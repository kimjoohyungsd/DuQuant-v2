"""Block-wise (per-DecoderLayer-output) training of a learnable SVD affine
transform per Linear layer, for MXFP4 W4A4 -- a FlatQuant-style alternative
to quantize/duquant.py's greedy block rotation.

Ported from Rotate-Test/flatquant/train_utils.py::cali_flat_quant (the
FlatQuant paper's own repo -- this conda env's namesake) as closely as
possible:
  - per decoder layer, sequentially (GPTQ/OmniQuant-style block calibration)
  - the FP teacher output of that layer (over the *unquantized* model) is
    computed first with weight_quant=act_quant=False (this repo's own
    passthrough, used in place of Rotate-Test's `_ori_mode` flag)
  - only the SVDGroupTransMatrix parameters are trained (AdamW + cosine
    annealing, self-normalized MSE loss `loss / loss.detach()` exactly as
    Rotate-Test does it, so every layer's loss starts at 1.0 regardless of
    that layer's output scale)
  - teacher-forced: the NEXT layer calibrates against this layer's FP
    output, not the just-trained quantized layer's own output (avoids
    compounding quantization error across the calibration sweep)

Unlike quantize/duquant.py, there is no rotation/permutation/smoothing/LET
here -- see quantize/svd_trans.py for the transform itself and
quantize/int_linear.py's `svd_trans` hook for how it reaches weight+activation.
"""
import copy
import gc
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn

from quantize.int_linear import QuantLinear
from quantize.svd_trans import SVDGroupTransMatrix
from quantize.utils import set_quant_state, register_scales_and_zeros

# UniformAffineQuantizer.per_token_fp4 hard-codes group_size=32 for the actual
# MXFP4 E2M1+E8M0 fake-quant regardless of --block_size (that flag is a
# quantize/duquant.py-specific rotation-search granularity, unrelated here).
# The SVD affine transform is sized to match the real MXFP4 group, per the
# brief ("MXFP4의 Group size의 맞게"), not to --block_size.
MXFP4_GROUP_SIZE = 32


def _get_decoder_layer_class(args):
    if "qwen" in args.net.lower():
        from models.int_qwen_layer import QuantQwenDecoderLayer
        return QuantQwenDecoderLayer
    # llama / vicuna / mistral all share the same wrapper in this repo
    from models.int_llama_layer import QuantLlamaDecoderLayer
    return QuantLlamaDecoderLayer


def _attach_svd_trans(qlayer, dev):
    """One SVDGroupTransMatrix per QuantLinear ("하나의 LinearLayer당 하나의
    Affine Matrix") -- not shared across q/k/v/gate/up the way FlatQuant's
    ln_trans/up_gate_trans are, per this port's brief."""
    for module in qlayer.modules():
        if isinstance(module, QuantLinear):
            module.svd_trans = SVDGroupTransMatrix(MXFP4_GROUP_SIZE).to(dev)


def _svd_trans_parameters(qlayer):
    params = []
    for n, p in qlayer.named_parameters():
        if "svd_trans" in n:
            params.append(p)
    return params


def _release_transient_quantizer_state(qlayer):
    """UniformAffineQuantizer.scale / .round_zero_point / .recorded_x_max are
    plain Python attributes (not nn.Parameter/register_buffer), assigned
    fresh in forward() every call -- so nn.Module.to()/.half()/.float() never
    sees or moves them. Left alone, a quantizer that has run at least one
    forward pass keeps pinning a GPU tensor even after the *module* is moved
    to CPU, and since every layer's trained QuantLlamaDecoderLayer is kept
    alive afterward (assigned back into model.model.layers for eval), this
    otherwise leaks GPU memory linearly across the calibration sweep -- see
    the investigation in this port's PR description.

    weight_quantizer's scale/round_zero_point are made real (moving) buffers
    via register_scales_and_zeros (matches quantize/duquant.py's own
    convention, and is what makes the *trained* weight scale actually used at
    eval time instead of silently recomputed). act_quantizer's are legitimately
    transient (per-token, recomputed every real forward pass including at
    eval) so they are just freed rather than persisted.

    CRITICAL: weight_quantizer.scale/round_zero_point were computed by the
    LAST *training* forward pass, i.e. from a chain of ops on the weight
    AFTER svd_trans -- a trainable parameter -- so they carry a live
    `grad_fn` back through that entire forward pass's graph (every
    intermediate activation, not just the ~KB scale tensor itself).
    register_buffer does not detach, so without the explicit .detach() below,
    each trained QuantLinear permanently pins its whole last training step's
    computation graph via this "buffer" for as long as the layer is kept
    alive (i.e. forever, since it's kept for eval) -- this was the actual
    multi-hundred-MB-per-layer leak (confirmed via gc.get_objects(): the
    reshape(-1, 32) MXFP4 activation tensors from training were still alive,
    reachable only through this un-detached buffer).
    """
    for module in qlayer.modules():
        if isinstance(module, QuantLinear):
            wq = module.weight_quantizer
            if wq.scale is not None:
                wq.scale = wq.scale.detach()
            if wq.round_zero_point is not None:
                wq.round_zero_point = wq.round_zero_point.detach()
    register_scales_and_zeros(qlayer)
    for module in qlayer.modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.recorded_x_max = None
            if module.act_quantizer is not None:
                module.act_quantizer.scale = None
                module.act_quantizer.round_zero_point = None
                module.act_quantizer.recorded_x_max = None


def blockwise_flatquant(lm, args, dataloader, logger):
    logger.info("Starting blockwise_flatquant calibration...")
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False

    DecoderLayer = _get_decoder_layer_class(args)
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)
    if args.deactive_amp:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros((args.nsamples, lm.seqlen, model.config.hidden_size),
                       dtype=dtype, device=dev)
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs.get("position_embeddings", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    layers[0] = layers[0].module.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1, 1).float()
    else:
        logger.info("No attention mask caught from the first layer.")
        attention_mask_batch = None
    position_ids = cache["position_ids"]
    position_embeddings = cache.get("position_embeddings", None)

    fp_inps = inps                       # teacher input to layer i
    fp_outs = torch.zeros_like(inps)     # teacher output of layer i -> teacher input to layer i+1
    loss_func = torch.nn.MSELoss()

    flat_parameters = {}
    for i in range(len(layers)):
        for name in ['q', 'k', 'v', 'gate', 'up', 'down', 'o']:
            exec(f"args.{name}_weight_quant_params = copy.copy(args.weight_quant_params)")
            exec(f"args.{name}_act_quant_params = copy.copy(args.act_quant_params)")

        logger.info(f"========= Layer {i} =========")
        layer = layers[i]
        qlayer = DecoderLayer(lm.model.config, layer, args, i).to(dev)
        _attach_svd_trans(qlayer, dev)

        # 1) FP teacher output over the *unquantized* layer (weight_quant=
        #    act_quant=False -> QuantLinear falls back to plain self.weight
        #    / no act_quantizer call, i.e. this repo's own "_ori_mode").
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        with torch.no_grad():
            with traincast():
                for j in range(args.nsamples):
                    fp_outs[j] = qlayer(fp_inps[j].unsqueeze(0),
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        position_embeddings=position_embeddings)[0]

        # 2) train only the SVD trans matrices to reconstruct that output
        #    through the quantized (weight+act) forward path.
        # NOTE: unlike Rotate-Test's cali_flat_quant, this does NOT upcast the
        # whole layer to fp32 before training -- the ~200M frozen weight
        # buffers stay in their native fp16 (halving their footprint), and
        # only the actually-trained SVDGroupTransMatrix parameters need fp32
        # (already true: nn.Linear/nn.Parameter default to fp32, untouched by
        # this). svd_trans.forward() casts its matrix to the input's dtype
        # before the matmul, so this is standard mixed-precision training
        # (fp16 compute, fp32 master params for the trained tensors) under
        # `traincast()`'s AMP autocast below -- gradients still flow back to
        # the fp32 parameters correctly through that cast.
        set_quant_state(qlayer, weight_quant=True, act_quant=True)
        for p in qlayer.parameters():
            p.requires_grad = False
        trained_params = _svd_trans_parameters(qlayer)
        for p in trained_params:
            p.requires_grad = True

        optimizer = torch.optim.AdamW(trained_params, lr=args.flat_lr)
        steps_per_epoch = max(1, args.nsamples // args.batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * steps_per_epoch, eta_min=args.flat_lr * 1e-3)

        for epoch in range(args.epochs):
            mse = 0.0
            tick = time.time()
            for j in range(steps_per_epoch):
                index = j * args.batch_size
                with traincast():
                    quant_out = qlayer(fp_inps[index:index + args.batch_size],
                                       attention_mask=attention_mask_batch,
                                       position_ids=position_ids,
                                       position_embeddings=position_embeddings)[0]
                    loss = loss_func(fp_outs[index:index + args.batch_size], quant_out)
                if not math.isfinite(loss.item()):
                    logger.info(f"layer {i} epoch {epoch} step {j}: loss is NaN/Inf, skipping step")
                    optimizer.zero_grad()
                    continue
                mse += loss.detach().float().cpu().item()
                loss = loss / loss.clone().detach()  # FlatQuant's self-normalized loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                del quant_out, loss
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(f"layer {i} epoch {epoch}, lr {cur_lr:.8f} "
                       f"time {time.time() - tick:.2f}s mse {mse:.8f}")

        flat_parameters[i] = {
            n: p.detach().cpu()
            for n, p in qlayer.named_parameters() if "svd_trans" in n
        }
        if args.save_dir:
            torch.save(flat_parameters, os.path.join(args.save_dir, "flat_parameters.pth"))

        for p in qlayer.parameters():
            p.requires_grad = False
        # see docstring: without this, every layer leaks the GPU tensors its
        # quantizers computed in forward() (plain attributes, invisible to
        # .to()), because qlayer is kept alive afterward (assigned into
        # model.model.layers below, for eval).
        _release_transient_quantizer_state(qlayer)

        # propagate the FP teacher's own activations forward (not this
        # layer's quantized output) -- matches Rotate-Test's calibration.
        layers[i] = qlayer.to("cpu")
        del layer, qlayer, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        fp_inps, fp_outs = fp_outs, fp_inps

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model
