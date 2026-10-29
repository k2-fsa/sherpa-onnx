#!/usr/bin/env python3
#
# Single-pass (non-autoregressive) decoder wrapper for Qwen3-ForcedAligner
# ONNX export. No KV cache: one forward produces logits[B, S, classify_num].

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import operators as onnx_ops


def _get_first_attr(obj, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            try:
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        v = int(v.item())
                    else:
                        continue
                return v
            except Exception:
                continue
    return default


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope_llama(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: (B, H, S, D); cos/sin: (B, S, D)
    dtype = x.dtype
    cos = cos.to(dtype=dtype).unsqueeze(1)
    sin = sin.to(dtype=dtype).unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbeddingFallback(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, rope_scaling=None):
        super().__init__()
        self.head_dim = int(head_dim)
        self.base = float(base)
        self.rope_scaling = rope_scaling

        if self.head_dim % 2 != 0:
            raise RuntimeError(
                f"RoPE requires even head_dim, got {self.head_dim}"
            )

        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, half, dtype=torch.float32) / half)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, S) -> cos/sin: (B, S, head_dim)
        pos = position_ids.to(torch.float32)
        freqs = torch.einsum("bs,d->bsd", pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb), torch.sin(emb)


class AlignerDecoderWrapper(nn.Module):
    # input_ids(B,S) int64, audio_features(B,A,H) f32, attention_mask(B,S) i64
    # -> logits(B,S,classify_num) f32
    def __init__(
        self,
        thinker: nn.Module,
        audio_token_id: int,
        hidden_size: int,
    ):
        super().__init__()
        self.thinker = thinker
        self.audio_token_id = int(audio_token_id)
        self.hidden_size = int(hidden_size)

        self.core = getattr(thinker, "model", None) or getattr(
            thinker, "core", None
        )
        if self.core is None:
            raise RuntimeError("Cannot find thinker.model/core")
        self.layers = self.core.layers
        self.norm = self.core.norm
        self.lm_head = getattr(thinker, "lm_head", None)
        if self.lm_head is None:
            raise RuntimeError("Cannot find lm_head on thinker")

        self.embed_tokens = getattr(self.core, "embed_tokens", None)
        if self.embed_tokens is None:
            raise RuntimeError("Cannot find embed_tokens")

        cfg = getattr(thinker, "config", None)
        if cfg is not None and hasattr(cfg, "text_config"):
            cfg = cfg.text_config

        self.num_heads = int(
            _get_first_attr(
                cfg, ["num_attention_heads", "num_heads", "n_head"], 0
            )
        )
        if self.num_heads <= 0:
            attn0 = self.layers[0].self_attn
            self.num_heads = int(
                _get_first_attr(attn0, ["num_heads", "n_heads"], 0)
            )

        attn0 = self.layers[0].self_attn
        q_out = int(attn0.q_proj.weight.shape[0])
        self._head_dim = int(q_out // self.num_heads)

        k_out = int(attn0.k_proj.weight.shape[0])
        self._num_kv_heads = int(k_out // self._head_dim)

        self.group_size = int(self.num_heads // self._num_kv_heads)
        self.qkv_dim = int(self.num_heads * self._head_dim)

        rope_theta = float(_get_first_attr(cfg, ["rope_theta"], 10000.0))
        rope_scaling = _get_first_attr(cfg, ["rope_scaling"], None)
        self.rope_fallback = RotaryEmbeddingFallback(
            self._head_dim, base=rope_theta, rope_scaling=rope_scaling
        )

    def _apply_qk_norm_if_any(
        self, attn_mod: nn.Module, q: torch.Tensor, k_kv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_norm = getattr(attn_mod, "q_norm", None)
        k_norm = getattr(attn_mod, "k_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k_kv = k_norm(k_kv)
        return q, k_kv

    def _inject_audio_features(
        self,
        tok: torch.Tensor,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        B, S, H = tok.shape
        mask = input_ids == self.audio_token_id
        m64 = mask.to(torch.int64)
        rank = torch.cumsum(m64, dim=1) - 1

        a_shape = onnx_ops.shape_as_tensor(audio_features)
        A1 = torch.clamp(a_shape[1].to(torch.int64) - 1, min=0)

        rank0 = torch.clamp(rank, min=0)
        rankc = torch.minimum(rank0, A1)

        idx = rankc.unsqueeze(-1).expand(B, S, H)
        gathered = torch.gather(audio_features, dim=1, index=idx)
        return torch.where(mask.unsqueeze(-1), gathered.to(tok.dtype), tok)

    def _attn_full(
        self,
        attn_mod: nn.Module,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = attn_mod.q_proj(x)
        k = attn_mod.k_proj(x)
        v = attn_mod.v_proj(x)

        q = q.view(B, S, self.num_heads, self._head_dim).permute(0, 2, 1, 3)
        k = k.view(B, S, self._num_kv_heads, self._head_dim).permute(
            0, 2, 1, 3
        )
        v = v.view(B, S, self._num_kv_heads, self._head_dim).permute(
            0, 2, 1, 3
        )

        q, k = self._apply_qk_norm_if_any(attn_mod, q, k)

        cos, sin = self.rope_fallback(position_ids)
        q = _apply_rope_llama(q, cos, sin)
        k = _apply_rope_llama(k, cos, sin)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scaling = getattr(attn_mod, "scaling", None)
        try:
            scale = (
                float(scaling)
                if scaling is not None
                else float(self._head_dim) ** -0.5
            )
        except Exception:
            scale = float(self._head_dim) ** -0.5

        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale

        q_pos = position_ids.unsqueeze(2)  # (B,S,1): query index i
        k_pos = position_ids.unsqueeze(1)  # (B,1,S): key index j
        causal = (k_pos <= q_pos).unsqueeze(1)  # (B,1,S,S): j <= i
        pad = attention_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
        keep = causal & (pad != 0)

        neg = torch.tensor(-1e4, dtype=scores.dtype, device=scores.device)
        scores = torch.where(keep, scores, neg)

        attn = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).reshape(B, S, self.qkv_dim)
        return attn_mod.o_proj(out)

    def _mlp(self, mlp_mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
        gate = mlp_mod.gate_proj(x)
        up = mlp_mod.up_proj(x)
        h = F.silu(gate) * up
        return mlp_mod.down_proj(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        model_dtype = next(self.thinker.parameters()).dtype

        tok = self.embed_tokens(input_ids).to(model_dtype)
        tok = self._inject_audio_features(
            tok, input_ids, audio_features.to(model_dtype)
        )

        position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids = torch.clamp(position_ids, min=0)

        x = tok
        for layer in self.layers:
            residual = x
            x_norm = layer.input_layernorm(x)
            attn_out = self._attn_full(
                layer.self_attn, x_norm, attention_mask, position_ids
            )
            x = residual + attn_out

            residual = x
            x_norm2 = layer.post_attention_layernorm(x)
            mlp_out = self._mlp(layer.mlp, x_norm2)
            x = residual + mlp_out

        x = self.norm(x)
        logits = self.lm_head(x).float()
        return logits
