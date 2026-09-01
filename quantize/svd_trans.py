import torch
import torch.nn as nn

"""FlatQuant-style learnable affine transform, sized to the MXFP4 group.

Ported from Rotate-Test/flatquant/trans_utils.py::SVDSingleTransMatrix (the
ICML'25 FlatQuant paper's own repo -- this conda env's namesake). FlatQuant
only ever uses that *undecomposed* SVD parameterization for small matrices
(o_trans: size=num_heads, kcache/vcache trans: size=head_dim) because its
Cayley orthogonal parametrization recomputes a full [size, size] matrix
inverse-like solve every forward pass -- O(size^3), fine at 32/128, but
prohibitive at hidden_size (4096: FlatQuant always Kronecker-decomposes
those). Here `size` is deliberately the MXFP4 group size (32), matching
UniformAffineQuantizer.per_token_fp4's fixed group_size=32 and DuQuant's own
block_size=32 convention -- see quantize/quantizer.py's `rotation()` /
--diverse_rotation for the discrete-search analogue this replaces with a
gradient-trained one.

T = U @ diag(s) @ V^T with U, V orthogonal (so T^-T = U @ diag(1/s) @ V^T --
no matrix inverse needed, just invert the diagonal). The SAME T is reused
block-diagonally for every group along the transformed dimension (like
DuQuant's shared, non-diverse rotation) -- there is exactly one of these per
QuantLinear ("하나의 LinearLayer당 하나의 Affine Matrix"), applied as
X' = X @ T to the activation and W' = W @ T^{-T} to the matching weight
columns, which preserves X @ W^T = X' @ W'^T exactly (pre-quantization) for
any invertible T, orthogonal or not -- same invariant DuQuant's rotation
relies on, just with a continuous, gradient-trained T instead of a
discrete, greedy-searched orthogonal one.
"""


class SVDGroupTransMatrix(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        u = nn.Linear(size, size, bias=False)
        nn.init.orthogonal_(u.weight)
        self.linear_u = nn.utils.parametrizations.orthogonal(
            u, orthogonal_map="cayley", use_trivialization=False)
        v = nn.Linear(size, size, bias=False)
        nn.init.orthogonal_(v.weight)
        self.linear_v = nn.utils.parametrizations.orthogonal(
            v, orthogonal_map="cayley", use_trivialization=False)
        self.diag = nn.Parameter(torch.ones(size))

    def get_matrix(self, inv_t: bool = False) -> torch.Tensor:
        diag = (1.0 / self.diag) if inv_t else self.diag
        return self.linear_u.weight @ torch.diag(diag) @ self.linear_v.weight.t()

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        orig_shape = x.shape
        matrix = self.get_matrix(inv_t=inv_t).to(x)
        x = x.reshape(-1, self.size).matmul(matrix)
        return x.reshape(orig_shape)
