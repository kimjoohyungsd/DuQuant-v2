import copy
import inspect
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from collections import OrderedDict
from models.transformation import *
from quantize.du_norm import DuQwenRMSNorm
from quantize.int_linear import QuantLinear

# apply_rotary_pos_emb / repeat_kv are byte-identical across llama, qwen2 and qwen3,
# so we take them from qwen2, which exists in every transformers that ships Qwen at
# all. This keeps the layer importable even on installs without a `qwen3` module --
# only *loading* a Qwen3 checkpoint needs transformers >= 4.51.
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.activations import ACT2FN


class QuantQwenMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
    ):
        super().__init__()
        self.gate_proj = QuantLinear(org_module.gate_proj,
                                     args.gate_weight_quant_params,
                                     args.gate_act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj,
                                     args.down_weight_quant_params,
                                     args.down_act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj,
                                   args.up_weight_quant_params,
                                   args.up_act_quant_params)
        self.act_fn = ACT2FN[hidden_act]
        self.init_duquant_params = torch.tensor(
            0) if args.gate_weight_quant_params[
                'quant_method'] in ('duquant', 'torq') else torch.tensor(1)

    def forward(self, x):
        if not self.init_duquant_params:
            # gate_proj and up_proj see the same input, so the second one reuses the
            # rotation/permutation the first one just calibrated (same as llama).
            act = self.act_fn(self.gate_proj(x))
            self.up_proj.copy_quantizers_duquant_params(self.gate_proj)
            out = self.down_proj(act * self.up_proj(x))
            self.init_duquant_params = torch.tensor(1)
            return out
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantQwenAttention(nn.Module):
    """Qwen2 / Qwen3 multi-head attention with DuQuant fake-quant projections.

    Qwen3 differs from Qwen2 in two ways that matter here:
      * q_proj/k_proj/v_proj carry no bias (config.attention_bias is False), and
      * every head of q and k is RMS-normalised by `q_norm`/`k_norm` *before* RoPE.
    Both are detected from the original module rather than from a version check, so
    the same class serves Qwen2, Qwen2.5 and Qwen3.
    """

    def __init__(self, org_module: nn.Module, config, args=None, layer_idx=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Qwen3 allows head_dim != hidden_size // num_heads, so trust the config.
        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_idx
        self.args = args

        # transformers < 4.48 keeps rotary_emb on the attention module; newer versions
        # keep one on the model and hand (cos, sin) to the layer as position_embeddings.
        self.rotary_emb = copy.deepcopy(getattr(org_module, "rotary_emb", None))

        # Qwen3 per-head q/k RMSNorm. Kept in full precision: it sits between the
        # projection and RoPE, is only head_dim wide, and is not a quantization target.
        self.q_norm = copy.deepcopy(getattr(org_module, "q_norm", None))
        self.k_norm = copy.deepcopy(getattr(org_module, "k_norm", None))

        self.k_proj = QuantLinear(org_module.k_proj,
                                  args.k_weight_quant_params,
                                  args.k_act_quant_params)
        self.v_proj = QuantLinear(org_module.v_proj,
                                  args.v_weight_quant_params,
                                  args.v_act_quant_params)
        self.q_proj = QuantLinear(org_module.q_proj,
                                  args.q_weight_quant_params,
                                  args.q_act_quant_params)
        self.o_proj = QuantLinear(org_module.o_proj,
                                  args.o_weight_quant_params,
                                  args.o_act_quant_params)

        self.use_weight_quant = False
        self.use_act_quant = False
        self.init_duquant_params = torch.tensor(
            0) if args.gate_weight_quant_params[
                'quant_method'] in ('duquant', 'torq') else torch.tensor(1)

    def _rope(self, value_states, position_ids, kv_seq_len, position_embeddings):
        if position_embeddings is not None:
            return position_embeddings
        if self.rotary_emb is None:
            raise ValueError(
                "No rotary embedding available: this transformers version keeps it on "
                "the model, so the decoder layer must forward `position_embeddings`.")
        if 'seq_len' in inspect.signature(self.rotary_emb.forward).parameters:
            return self.rotary_emb(value_states, seq_len=kv_seq_len)
        return self.rotary_emb(value_states, position_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        indices_k=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q/k/v share one input, so k and v reuse q's freshly calibrated transform.
        if not self.init_duquant_params:
            self.k_proj.copy_quantizers_duquant_params(self.q_proj)
        key_states = self.k_proj(hidden_states)
        if not self.init_duquant_params:
            self.v_proj.copy_quantizers_duquant_params(self.q_proj)
        value_states = self.v_proj(hidden_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"{self.__class__.__name__} needs a layer_idx for k/v caching.")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self._rope(value_states, position_ids, kv_seq_len,
                              position_embeddings)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx,
                {"sin": sin, "cos": cos})

        key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        # Qwen3: normalise each head of q and k before RoPE.
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        _rope_params = inspect.signature(apply_rotary_pos_emb).parameters
        if 'unsqueeze_dim' in _rope_params and 'position_ids' not in _rope_params:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size "
                f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}")

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                    f"but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2)
        # num_heads * head_dim, not hidden_size: Qwen3 lets these differ.
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        self.init_duquant_params = torch.tensor(1)
        return attn_output, attn_weights, past_key_value

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantLinear):
                m.set_quant_state(weight_quant, act_quant)


class QuantQwenDecoderLayer(nn.Module):
    def __init__(self, config, ori_layer, args, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantQwenAttention(org_module=ori_layer.self_attn,
                                            config=config,
                                            args=args,
                                            layer_idx=layer_idx)
        self.mlp = QuantQwenMLP(
            org_module=ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            args=args,
        )
        self.input_layernorm = DuQwenRMSNorm(
            ori_layer.input_layernorm,
            eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = DuQwenRMSNorm(
            ori_layer.post_attention_layernorm,
            eps=ori_layer.post_attention_layernorm.variance_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        indices_k=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).half()
        hidden_states = self.mlp(
            hidden_states.to(self.mlp.up_proj.weight.device)).to(residual.device)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )
        if output_attentions:
            outputs += (self_attn_weights, )
        if use_cache:
            outputs += (present_key_value, )
        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for name, m in self.named_modules():
            if isinstance(m, QuantLinear):
                m.set_quant_state(weight_quant, act_quant)

    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.input_layernorm, [
                self.self_attn.q_proj, self.self_attn.k_proj,
                self.self_attn.v_proj
            ], self.qkv_smooth_scale, self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.post_attention_layernorm,
                                    [self.mlp.up_proj, self.mlp.gate_proj],
                                    self.fc1_smooth_scale,
                                    self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj,
                                   self.self_attn.o_proj,
                                   self.out_smooth_scale,
                                   self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                 self.qkt_smooth_scale)
            self.mlp.down_proj.temp_weight = self.mlp.down_proj.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(
                        module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter = True

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.input_layernorm, [
                self.self_attn.q_proj, self.self_attn.k_proj,
                self.self_attn.v_proj
            ], self.qkv_smooth_scale, self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.post_attention_layernorm,
                                  [self.mlp.up_proj, self.mlp.gate_proj],
                                  self.fc1_smooth_scale, self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj, self.self_attn.o_proj,
                                 self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                               self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter = False

    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)

    def duquant_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)

    def duquant_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix +
                            name] = param if keep_vars else param.detach()
        return destination

    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()

    def register_duquant_params(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantQwenMLP) or isinstance(
                    module, QuantQwenAttention):
                delattr(module, 'init_duquant_params')
                module.register_buffer('init_duquant_params', torch.tensor(1))
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_duquant_params()
                module.act_quantizer.register_duquant_params()

    def load_duquant_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('R') > -1 or k.find('permutation_list') > -1 or k.find(
                    'init_duquant_params') > -1:
                exec(f'self.{k} = v.to(device)')

    def load_smooth_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('smooth') > -1:
                # exec(f'self.{k} = v')
                self.register_parameter(
                    k, torch.nn.Parameter(v.to(device), requires_grad=False))

    def load_post_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('post') > -1:
                # exec(f'self.{k} = v')
                rg = False if k.find('down') > -1 else True
                self.register_parameter(
                    k, torch.nn.Parameter(v.to(device), requires_grad=rg))

    def load_lwc_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('bound_factor') > -1:
                v = torch.nn.Parameter(v.to(device))
                exec(f'self.{k} = v.to(device)')
