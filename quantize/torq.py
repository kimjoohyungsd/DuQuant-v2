# coding=utf-8
"""TORQ (Two-level Orthogonal Rotation for MXFP4 Quantization), Xu & Hu et al.
2026, arXiv:2605.19561 -- ported as a new --quant_method, alongside duquant
(greedy block rotation) and hadamard (fixed randomized block rotation).

Where duquant SEARCHES a rotation that clusters outlier channels together and
hadamard uses a FIXED data-independent rotation, TORQ CALIBRATES two rotations
from calibration activations, each attacking a different failure mode of MXFP4
(E2M1 element codebook + per-block E8M0 shared exponent, block size K):

  R_inter in O(B)  (B = hidden_size // K blocks per token): "Macro-Equilibrium
    Rotation". Equalizes each block's energy (sum of squares) across blocks so
    a handful of high-energy blocks can't force every block's shared exponent
    up and zero out the small-magnitude elements sharing that exponent
    (Sec 4.1, Theorem 4.1, Algorithm 1).

  R_intra in O(K): "Micro-Alignment Rotation". Within a block, redistributes
    values across the 8 positive E2M1 codewords {0,.5,1,1.5,2,3,4,6} so they
    are used close to uniformly instead of piling up near zero ("codebook
    collapse"), via alternating S-step (per-block E8M0 scale) / R-step
    (Givens rotations minimizing a codeword-occupancy-imbalance loss)
    (Sec 4.2, Algorithm 2).

Applied online at every forward (both weight and activation quantizers, via
UniformAffineQuantizer.init_duquant -- see quantize/quantizer.py), exactly as:

    X_reshaped [.., B, K] --R_inter (left, mixes across blocks)--> Y
    Y --R_intra (right, mixes within a block)--> Z  (quantized from here)

matching the paper's Algorithm 3. Because both the weight and the matching
activation quantizer apply the *same pair* (R_inter, R_intra) -- calibrated
once from the layer's actual input and shared via
UniformAffineQuantizer.copy_duquant_params, exactly like duquant's R -- X @ W^T
is preserved before quantization: rotating both operands by the same
orthogonal transform on the contracted (hidden) axis leaves their dot product
unchanged (R_inter R_inter^T = I, R_intra R_intra^T = I).

Design choice vs. the paper (documented, not a bug): Sec 4.1.2 / Appendix A.7
describe estimating a *separate* R_inter per within-block position k (K
covariance matrices Sigma_k, K independent Algorithm-1 runs). Algorithm 3
(the actual inference recipe) and the weight-fusion story
(W' = W (R_intra^T (x) R_inter^T), a SINGLE Kronecker product) only work with
one shared R_inter and one shared R_intra -- a per-position family of R_inter
matrices has no such closed-form fusion. This port follows Algorithm 3: ONE
R_inter (from a single B x B covariance of per-block energy, pooled over all K
positions and all calibration tokens) and ONE R_intra per calibrated layer.
"""
import math

import torch


# MXFP4 (e2m1) positive codewords {0, .5, 1, 1.5, 2, 3, 4, 6}, J = 8. Decision
# boundaries are the same round-to-nearest midpoints quantize/fp4_ops.py's
# cast_to_fp4 already hardcodes (0.25, 0.75, ..., 5.0) -- kept as a module
# constant here so the codebook-occupancy loss uses the exact same bins the
# quantizer will actually round to.
FP4_POS_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # len J-1 = 7
FP4_J = len(FP4_POS_LEVELS)  # 8
FP4_CMAX = FP4_POS_LEVELS[-1]  # 6.0


# ---------------------------------------------------------------------------
# 4.1 Macro-Equilibrium Rotation (inter-block)
# ---------------------------------------------------------------------------

def _givens_variance_equalize(Sigma: torch.Tensor, eps: float = 1e-4,
                              max_iter: int = 2000) -> torch.Tensor:
    """Algorithm 1: Givens Rotation for Variance Equalization.

    Input: symmetric PSD Sigma [B, B] (float32/float64). Output: orthogonal
    R [B, B] such that diag(R^T Sigma R) ~= (tr(Sigma)/B) * 1 -- i.e. R is
    built up right-multiplying (R <- R @ G), so applying it as Y = R^T X
    (see apply_inter_block_rotation) equalizes X's per-row variance, matching
    Theorem 4.1's diag(R Sigma R^T) convention up to this transpose.

    max_iter is a safety cap the paper's pseudocode omits (pure numerical
    convergence isn't guaranteed to hit `eps` exactly on real data); once hit,
    R is returned as-is -- still a valid orthogonal matrix, just not fully
    equalized.
    """
    B = Sigma.shape[0]
    Sigma = Sigma.clone()
    R = torch.eye(B, dtype=Sigma.dtype, device=Sigma.device)
    c = torch.diagonal(Sigma).sum() / B

    for _ in range(max_iter):
        diag = torch.diagonal(Sigma)
        dev = diag - c
        worst = dev.abs().argmax()
        if dev[worst].abs() <= eps:
            break
        # "block pair (i, j) with opposite variance deviation signs": among
        # all j with dev[worst]*dev[j] < 0, take the most-opposite one so
        # each Givens step makes maximal progress.
        i = worst.item()
        candidates = (dev * dev[i] < 0).nonzero(as_tuple=True)[0]
        if candidates.numel() == 0:
            # no eligible partner (shouldn't happen while sum(dev) == 0 and
            # dev[i] != 0, but numerical drift can zero out the rest) -- stop.
            break
        j = candidates[dev[candidates].abs().argmax()].item()

        sigma_ii, sigma_jj, sigma_ij = Sigma[i, i], Sigma[j, j], Sigma[i, j]
        # NOTE: the paper's Eq. in Algorithm 1 literally reads
        # theta = 0.5*atan2(2*sigma_ij, sigma_ii-sigma_jj) -- that is the
        # classic Jacobi *diagonalizing* angle (drives sigma_ij -> 0, which
        # pushes the two diagonal entries APART to the 2x2 block's
        # eigenvalues, the opposite of equalizing them). Solving
        # sigma_ii' = sigma_jj' directly instead gives
        # tan(2*theta) = (sigma_jj - sigma_ii) / (2*sigma_ij); verified
        # analytically and against a diagonal-Sigma synthetic test (paper's
        # formula degenerates to theta=0 whenever sigma_ij=0, i.e. never
        # equalizes already-uncorrelated blocks, which this formula does).
        theta = 0.5 * torch.atan2(sigma_jj - sigma_ii, 2 * sigma_ij)
        cth, sth = torch.cos(theta), torch.sin(theta)

        # G is the identity except its (i,i),(i,j),(j,i),(j,j) 2x2 block ->
        # apply Sigma <- G^T Sigma G and R <- R G directly on rows/cols i,j
        # (equivalent to, and far cheaper than, materializing G).
        col_i, col_j = Sigma[:, i].clone(), Sigma[:, j].clone()
        Sigma[:, i] = cth * col_i + sth * col_j
        Sigma[:, j] = -sth * col_i + cth * col_j
        row_i, row_j = Sigma[i, :].clone(), Sigma[j, :].clone()
        Sigma[i, :] = cth * row_i + sth * row_j
        Sigma[j, :] = -sth * row_i + cth * row_j

        r_i, r_j = R[:, i].clone(), R[:, j].clone()
        R[:, i] = cth * r_i + sth * r_j
        R[:, j] = -sth * r_i + cth * r_j

    return R


def compute_r_inter(acts: torch.Tensor, block_size: int, eps: float = 1e-4,
                    max_iter: int = 2000, damping: float = 1e-8) -> torch.Tensor:
    """Calibrate the single shared R_inter in O(B) from calibration
    activations `acts` [N, d] (any leading dims are flattened into N).

    Sigma[b, b'] = E_{n,k}[ X[n,b,k] * X[n,b',k] ]: the true second-moment
    (uncentered covariance) of the block-index axis, pooled over every
    within-block position k and every calibration sample n. This is the
    quantity Y = R_inter^T X actually transforms linearly -- for any k,
    Y[n,a,k] = sum_b R[b,a] X[n,b,k], so E_{n,k}[Y[n,a,k]^2] =
    (R^T Sigma R)[a,a] exactly -- so equalizing diag(R^T Sigma R) (Algorithm
    1) really does equalize each block's expected per-element squared
    magnitude after rotation. (An earlier version of this function used
    Sigma = Cov(per-sample block ENERGY vector e_n[b] = ||z_{n,b}||^2, i.e.
    already reduced over k) -- energy is a quadratic function of X, so
    rotating X by R does *not* transform that energy-covariance's diagonal
    the same way; the fix above matches Theorem 4.1's actual linear-algebra
    argument, and its O(N*K*B^2) cost matches Appendix A.7's stated
    O(T*B^2*K) covariance-estimation complexity.)
    """
    acts = acts.reshape(-1, acts.shape[-1]).to(torch.float64)
    N, d = acts.shape
    assert d % block_size == 0, f"hidden dim {d} not a multiple of block_size {block_size}"
    B = d // block_size
    # [N, B, K] -> [N, K, B] -> [N*K, B]: one row per (sample, position) pair,
    # each holding that position's value across all B blocks.
    flat = acts.reshape(N, B, block_size).permute(0, 2, 1).reshape(-1, B)
    Sigma = (flat.t() @ flat) / max(flat.shape[0], 1)
    Sigma = Sigma + damping * torch.eye(B, dtype=Sigma.dtype, device=Sigma.device)
    R = _givens_variance_equalize(Sigma, eps=eps, max_iter=max_iter)
    return R.to(torch.float32)


def apply_inter_block_rotation(x: torch.Tensor, R_inter: torch.Tensor,
                               block_size: int) -> torch.Tensor:
    """Y = R_inter^T @ X per Algorithm 3 (X: [B, K] for one token; batched
    here over every leading dim of `x`). Mixes across blocks (same
    within-block position k, all B blocks) -- the complement of
    _apply_block_rotation in quantize/quantizer.py, which mixes *within* one
    block (duquant/hadamard's R_intra-only case)."""
    orig_shape = x.shape
    B = R_inter.shape[0]
    X = x.reshape(-1, B, block_size)
    R = R_inter.to(dtype=X.dtype, device=X.device)
    Y = torch.einsum('ba,nbk->nak', R, X)  # Y[n,a,k] = sum_b R[b,a] X[n,b,k] = (R^T X)[a,k]
    return Y.reshape(orig_shape)


# ---------------------------------------------------------------------------
# 4.2 Micro-Alignment Rotation (intra-block)
# ---------------------------------------------------------------------------

def _codeword_occupancy(mag: torch.Tensor) -> torch.Tensor:
    """mag: [.., N] nonnegative normalized magnitudes -> p_hat [.., J]
    empirical occupancy of each of the J=8 positive E2M1 codewords."""
    boundaries = torch.tensor(FP4_BOUNDARIES, dtype=mag.dtype, device=mag.device)
    idx = torch.bucketize(mag, boundaries)  # in [0, J-1]
    counts = torch.zeros(mag.shape[:-1] + (FP4_J,), dtype=mag.dtype, device=mag.device)
    counts.scatter_add_(-1, idx, torch.ones_like(mag))
    return counts / mag.shape[-1]


def _imbalance_scores(Z_norm: torch.Tensor) -> torch.Tensor:
    """Z_norm: [B, K] (normalized, i.e. already divided by each row's scale)
    -> h [K], Eq. 21: mean-squared distance of each column's codeword
    occupancy from uniform (1/J)."""
    p_hat = _codeword_occupancy(Z_norm.abs().t())  # [K, J]
    return ((p_hat - 1.0 / FP4_J) ** 2).sum(dim=-1)


def _select_column_pairs(Z_norm: torch.Tensor, k_top: int, num_pairs: int,
                         lam: float):
    """Appendix A.5: filter to the k_top most codebook-imbalanced columns,
    score all candidate pairs by imbalance + lambda * anti-correlation
    (Eq. 20-23), and greedily take the top `num_pairs` non-overlapping ones.
    Returns a list of (p, q) column-index pairs."""
    K = Z_norm.shape[1]
    p_hat = _codeword_occupancy(Z_norm.abs().t())  # [K, J]
    dev = p_hat - 1.0 / FP4_J  # [K, J]
    h = (dev ** 2).sum(dim=-1)  # [K]

    k_top = max(2, min(k_top, K))
    cand = torch.topk(h, k_top).indices  # column indices, candidate pool

    # H_{k,l} = h_k + h_l + lambda * c_{k,l},  c_{k,l} = -sum_j dev_k,j dev_l,j
    dev_c = dev[cand]  # [k_top, J]
    c_mat = -(dev_c @ dev_c.t())  # [k_top, k_top]
    H = h[cand].unsqueeze(1) + h[cand].unsqueeze(0) + lam * c_mat
    H.fill_diagonal_(-float("inf"))

    flat = torch.argsort(H.reshape(-1), descending=True)
    used = set()
    pairs = []
    for idx in flat.tolist():
        if len(pairs) >= num_pairs:
            break
        a, b = idx // k_top, idx % k_top
        if a == b or H[a, b] == -float("inf"):
            continue
        ka, kb = cand[a].item(), cand[b].item()
        if ka in used or kb in used:
            continue
        pairs.append((ka, kb))
        used.add(ka)
        used.add(kb)
    return pairs


def _pair_loss_batched(u: torch.Tensor, v: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """The two-column contribution to L_code (Eq. 5) after rotating (u, v) by
    each candidate angle in `thetas` [M] -- the only part of the global loss
    that changes when just this pair is rotated, so minimizing it minimizes
    L_code restricted to this pair's angle (see _exact_angle_solver).
    Vectorized over all M candidates at once: u, v: [S] -> loss: [M]."""
    c, s = torch.cos(thetas), torch.sin(thetas)  # [M]
    u_rot = u.unsqueeze(1) * c - v.unsqueeze(1) * s  # [S, M]
    v_rot = u.unsqueeze(1) * s + v.unsqueeze(1) * c  # [S, M]
    hu = _imbalance_scores(u_rot.t())  # [M]
    hv = _imbalance_scores(v_rot.t())  # [M]
    return hu + hv


def _exact_angle_solver(u: torch.Tensor, v: torch.Tensor, max_candidates: int = 4096) -> float:
    """Appendix A.4: exact global optimum of the (piecewise-constant)
    two-column codebook loss over theta in [0, 2*pi), via a critical-angle
    enumeration instead of gradient descent.

    Rotation convention (must match apply below): u' = u cos(t) - v sin(t),
    v' = u sin(t) + v cos(t), i.e. u' = r*cos(phi+t), v' = r*sin(phi+t) with
    r, phi the polar form of (u, v). |u'| crosses boundary d_j when
    cos(phi+t) = +/- d_j/r -> t = -phi +/- arccos(d_j/r); |v'| crosses when
    sin(phi+t) = +/- d_j/r -> t = pi/2 - phi +/- arccos(d_j/r).

    max_candidates caps the (u.numel() * 7 * 4) raw crossing count via random
    subsampling before the O(samples x candidates) batched loss eval below --
    a deliberate accuracy/speed tradeoff the paper's O(|T|) claim glosses
    over for realistic calibration sample counts.
    """
    r = torch.sqrt(u * u + v * v)
    phi = torch.atan2(v, u)
    boundaries = torch.tensor(FP4_BOUNDARIES, dtype=u.dtype, device=u.device)

    two_pi = 2 * math.pi
    crit = [torch.zeros(1, dtype=u.dtype, device=u.device)]  # theta=0 always included
    valid = r.unsqueeze(-1) >= boundaries  # [S, J-1]
    if valid.any():
        ratio = (boundaries / r.clamp(min=1e-12).unsqueeze(-1)).clamp(max=1.0)
        alpha = torch.arccos(ratio)  # [S, J-1]
        for base in (-phi.unsqueeze(-1), math.pi / 2 - phi.unsqueeze(-1)):
            for sign in (1.0, -1.0):
                t = (base + sign * alpha)[valid]
                if t.numel():
                    crit.append(t)
    angles = torch.cat(crit) % two_pi
    angles = torch.unique(angles)
    angles, _ = torch.sort(angles)
    if angles.numel() < 2:
        mids = angles
    else:
        wrapped_next = torch.cat([angles[1:], angles[:1] + two_pi])
        mids = (angles + wrapped_next) / 2 % two_pi
    if mids.numel() > max_candidates:
        keep = torch.randperm(mids.numel(), device=mids.device)[:max_candidates]
        mids = mids[keep]
    if mids.numel() == 0:
        return 0.0

    losses = _pair_loss_batched(u, v, mids)  # [M], one shot, no Python loop
    return mids[torch.argmin(losses)].item()


def _apply_givens_pair(R: torch.Tensor, p: int, q: int, theta: float) -> torch.Tensor:
    """R <- R @ G_{(p,q)}(theta), same 2x2-block convention as _pair_loss."""
    c = math.cos(theta)
    s = math.sin(theta)
    col_p, col_q = R[:, p].clone(), R[:, q].clone()
    R[:, p] = c * col_p - s * col_q
    R[:, q] = s * col_p + c * col_q
    return R


def compute_r_intra(acts_after_inter: torch.Tensor, block_size: int,
                    max_iter: int = 10, k_top_frac: float = 0.5,
                    num_pairs: int = None, lam: float = 1.0,
                    loss_eps: float = 1e-5, max_samples: int = 8192,
                    generator: torch.Generator = None) -> torch.Tensor:
    """Algorithm 2: alternating S-step / R-step construction of R_intra
    in O(K), from activations `acts_after_inter` [.., d] (already rotated by
    R_inter -- i.e. this must be called with the SAME activations
    apply_inter_block_rotation(x, R_inter, block_size) produced).

    Every (token, block) instance is pooled into one flat "B" of block
    samples (Sec 4.2 reuses the symbol B for this pooled sample count, not
    the per-token block count of Sec 4.1 -- see the module docstring).
    max_samples subsamples that pool for tractability (critical-angle search
    is O(samples) per pair per iteration).
    """
    K = block_size
    X = acts_after_inter.reshape(-1, K).to(torch.float32)
    if X.shape[0] > max_samples:
        idx = torch.randperm(X.shape[0], generator=generator)[:max_samples]
        X = X[idx]

    k_top = max(2, int(round(k_top_frac * K)))
    if num_pairs is None:
        num_pairs = max(1, k_top // 2)

    R_intra = torch.eye(K, dtype=torch.float32, device=X.device)
    prev_loss = None
    for _ in range(max_iter):
        rotated = X @ R_intra  # [samples, K]

        # S-step (Eq. 6): per-sample (per-block) E8M0-style power-of-2 scale
        # mapping that block's max onto the largest codeword.
        block_max = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        s = torch.pow(2.0, torch.floor(torch.log2(block_max / FP4_CMAX)))
        Z = rotated / s  # normalized ("Normalized FP4 Space")

        cur_loss = _imbalance_scores(Z).sum().item()
        if prev_loss is not None and abs(prev_loss - cur_loss) < loss_eps:
            break
        prev_loss = cur_loss

        pairs = _select_column_pairs(Z, k_top, num_pairs, lam)
        if not pairs:
            break
        for p, q in pairs:
            u, v = Z[:, p], Z[:, q]
            theta = _exact_angle_solver(u, v)
            R_intra = _apply_givens_pair(R_intra, p, q, theta)
            # keep Z consistent for the remaining pairs this iteration
            c, s2 = math.cos(theta), math.sin(theta)
            new_u = Z[:, p] * c - Z[:, q] * s2
            new_v = Z[:, p] * s2 + Z[:, q] * c
            Z = Z.clone()
            Z[:, p], Z[:, q] = new_u, new_v

    return R_intra


# ---------------------------------------------------------------------------
# End-to-end calibration entry point
# ---------------------------------------------------------------------------

def calibrate(acts: torch.Tensor, block_size: int, **kwargs):
    """Calibrate (R_inter, R_intra) from one layer's calibration activations
    `acts` (any shape [.., d], d = hidden_size). kwargs are split between
    compute_r_inter (inter_*) and compute_r_intra (the rest); see their
    signatures. Returns (R_inter [B,B] float32, R_intra [K,K] float32)."""
    inter_kwargs = {k[len("inter_"):]: v for k, v in kwargs.items() if k.startswith("inter_")}
    intra_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("inter_")}
    R_inter = compute_r_inter(acts, block_size, **inter_kwargs)
    y = apply_inter_block_rotation(acts, R_inter, block_size)
    R_intra = compute_r_intra(y, block_size, **intra_kwargs)
    return R_inter, R_intra


def apply_torq_rotation(x: torch.Tensor, R_inter: torch.Tensor, R_intra: torch.Tensor,
                        block_size: int) -> torch.Tensor:
    """Algorithm 3 steps 2-3: Y = R_inter^T X (across blocks), Z = Y R_intra
    (within a block). Used identically for weight and activation tensors."""
    y = apply_inter_block_rotation(x, R_inter, block_size)
    orig_shape = y.shape
    z = y.reshape(-1, block_size) @ R_intra.to(dtype=y.dtype, device=y.device)
    return z.reshape(orig_shape)
