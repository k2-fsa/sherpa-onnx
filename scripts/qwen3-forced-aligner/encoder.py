#!/usr/bin/env python3
#
# Audio encoder for Qwen3-ASR ONNX export (backend-only avoids Conv in ONNX).

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_frontend import _pick


def _infer_text_hidden(thinker: nn.Module) -> int:
    text_hidden = 0
    cfg = getattr(thinker, "config", None)
    if cfg is not None:
        tc = getattr(cfg, "text_config", None)
        if tc is None and hasattr(cfg, "thinker_config"):
            tc = getattr(
                getattr(cfg, "thinker_config", None), "text_config", None
            )
        if tc is not None:
            text_hidden = int(getattr(tc, "hidden_size", 0) or 0)
    if text_hidden <= 0:
        core = getattr(thinker, "model", None) or getattr(thinker, "core", None)
        if core is not None and hasattr(core, "embed_tokens"):
            text_hidden = int(core.embed_tokens.weight.shape[1])
    if text_hidden <= 0:
        raise RuntimeError("Cannot infer text hidden_size")
    return int(text_hidden)


def _find_audio_proj(root: nn.Module, in_dim: int, out_dim: int) -> nn.Module:
    cand_names = [
        "audio_feature_projection",
        "audio_projection",
        "audio_proj",
        "audio_projector",
        "audio_to_text_proj",
        "audio_to_text",
        "mm_projector",
        "proj_audio",
        "audio_linear",
        "audio_fc",
    ]

    for holder in (
        root,
        getattr(root, "model", None),
        getattr(root, "core", None),
    ):
        if holder is None:
            continue
        for name in cand_names:
            m = getattr(holder, name, None)
            if m is None:
                continue
            if (
                isinstance(m, nn.Linear)
                and int(m.in_features) == int(in_dim)
                and int(m.out_features) == int(out_dim)
            ):
                return m
            if isinstance(m, nn.Module):
                for sub in m.modules():
                    if (
                        isinstance(sub, nn.Linear)
                        and int(sub.in_features) == int(in_dim)
                        and int(sub.out_features) == int(out_dim)
                    ):
                        return sub

    for _, m in root.named_modules():
        if (
            isinstance(m, nn.Linear)
            and int(m.in_features) == int(in_dim)
            and int(m.out_features) == int(out_dim)
        ):
            return m

    raise RuntimeError(
        f"Cannot find audio_proj Linear({in_dim}->{out_dim}) in thinker"
    )


class EncoderBackend(nn.Module):
    def __init__(
        self,
        audio_tower: nn.Module,
        tokens_per_chunk: int,
        window_aftercnn: int,
    ):
        super().__init__()
        self.audio_tower = audio_tower

        self.layers = _pick(audio_tower, ["layers"])
        if self.layers is None:
            raise RuntimeError("Cannot find audio_tower.layers")

        self.positional_embedding = _pick(
            audio_tower,
            ["positional_embedding", "embed_positions", "position_embedding"],
        )
        if self.positional_embedding is None or not hasattr(
            self.positional_embedding, "positional_embedding"
        ):
            raise RuntimeError(
                "Cannot find positional_embedding.positional_embedding"
            )

        self.ln_post = _pick(
            audio_tower, ["ln_post", "layer_norm", "post_layernorm"]
        )
        if self.ln_post is None:
            raise RuntimeError("Cannot find ln_post")

        self.proj1 = _pick(audio_tower, ["proj1"])
        self.proj2 = _pick(audio_tower, ["proj2"])
        if self.proj1 is None or self.proj2 is None:
            raise RuntimeError("Cannot find proj1/proj2")

        self.tokens_per_chunk = int(tokens_per_chunk)
        self.window_aftercnn = int(max(1, window_aftercnn))

    def _attn_eager(
        self,
        attn_mod: nn.Module,
        x: torch.Tensor,
        key_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, T, _ = x.shape
        num_heads = int(
            getattr(attn_mod, "num_heads", getattr(attn_mod, "n_heads", 0)) or 0
        )
        head_dim = int(getattr(attn_mod, "head_dim", 0) or 0)
        if num_heads <= 0 or head_dim <= 0:
            q_out = int(attn_mod.q_proj.weight.shape[0])
            num_heads = max(1, num_heads)
            head_dim = int(q_out // num_heads)

        scaling = getattr(attn_mod, "scaling", None)
        try:
            scale = float(scaling) if scaling is not None else (head_dim**-0.5)
        except Exception:
            scale = head_dim**-0.5

        q = attn_mod.q_proj(x).view(B, T, num_heads, head_dim).transpose(1, 2)
        k = attn_mod.k_proj(x).view(B, T, num_heads, head_dim).transpose(1, 2)
        v = attn_mod.v_proj(x).view(B, T, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        if key_mask is not None:
            km = key_mask.to(dtype=scores.dtype).unsqueeze(1).unsqueeze(2)
            scores = scores + (1.0 - km) * (-1e4)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, num_heads * head_dim)

        proj = getattr(attn_mod, "out_proj", None)
        if proj is None:
            proj = getattr(attn_mod, "o_proj", None)
        out = proj(out)

        if key_mask is not None:
            out = out * key_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out

    def _layer_norm(
        self, layer: nn.Module, names: List[str], x: torch.Tensor
    ) -> torch.Tensor:
        m = _pick(layer, names)
        if m is None:
            raise RuntimeError(f"Cannot find layer norm in {names}")
        return m(x)

    def _mlp(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        fc1 = _pick(layer, ["fc1"])
        fc2 = _pick(layer, ["fc2"])
        if fc1 is None or fc2 is None:
            mlp = _pick(layer, ["mlp"])
            if mlp is None:
                raise RuntimeError("Cannot find mlp/fc1/fc2")
            gate = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            return mlp.down_proj(F.silu(gate) * up)

        h = fc1(x)
        act = _pick(layer, ["activation_fn"])
        h = F.gelu(h) if act is None else act(h)
        return fc2(h)

    def forward(
        self, hidden_states: torch.Tensor, token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, T, H = hidden_states.shape
        device = hidden_states.device

        tn = self.tokens_per_chunk
        pos_idx = torch.arange(T, device=device) % tn
        pos_emb = F.embedding(
            pos_idx, self.positional_embedding.positional_embedding
        )
        x = hidden_states + pos_emb.unsqueeze(0).to(dtype=hidden_states.dtype)

        if token_mask is not None:
            x = x * token_mask.unsqueeze(-1).to(dtype=x.dtype)

        # The HF reference passes cu_seqlens only to FlashAttention-2; under
        # eager/sdpa the audio encoder is full bidirectional attention over
        # all valid tokens, so we do the same here.
        km = token_mask

        for layer in self.layers:
            residual = x
            x_norm = self._layer_norm(
                layer,
                ["self_attn_layer_norm", "input_layer_norm", "input_layernorm"],
                x,
            )
            attn_mod = _pick(layer, ["self_attn", "attention"])
            x = residual + self._attn_eager(attn_mod, x_norm, km)

            residual = x
            x_norm2 = self._layer_norm(
                layer,
                [
                    "final_layer_norm",
                    "post_attention_layer_norm",
                    "post_attention_layernorm",
                ],
                x,
            )
            x = residual + self._mlp(layer, x_norm2)

            if km is not None:
                x = x * km.unsqueeze(-1).to(dtype=x.dtype)

        x = self.ln_post(x)
        x = self.proj1(x)
        x = F.gelu(x)
        x = self.proj2(x)
        return x


class AudioEncoderWrapper(nn.Module):
    def __init__(
        self, thinker: nn.Module, tokens_per_chunk: int, window_aftercnn: int
    ):
        super().__init__()
        audio_tower = getattr(thinker, "audio_tower", None)
        if audio_tower is None:
            raise RuntimeError("Cannot find thinker.audio_tower")

        self.text_hidden = _infer_text_hidden(thinker)
        self.tokens_per_chunk = int(tokens_per_chunk)
        self.window_aftercnn = int(max(1, window_aftercnn))

        self.backend = EncoderBackend(
            audio_tower,
            tokens_per_chunk=self.tokens_per_chunk,
            window_aftercnn=self.window_aftercnn,
        )

        out_dim = int(
            getattr(getattr(audio_tower, "config", None), "output_dim", 0) or 0
        )
        if out_dim <= 0:
            try:
                out_dim = int(self.backend.proj2.weight.shape[0])
            except Exception:
                out_dim = int(self.text_hidden)

        self.audio_proj: Optional[nn.Module] = None
        if int(out_dim) != int(self.text_hidden):
            self.audio_proj = _find_audio_proj(
                thinker, in_dim=int(out_dim), out_dim=int(self.text_hidden)
            )

    def forward(
        self, input_features: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        hs = self.backend(input_features, token_mask)

        if self.audio_proj is not None:
            hs = self.audio_proj(hs)

        hs = hs * token_mask.unsqueeze(-1).to(dtype=hs.dtype)
        return hs.to(torch.float32)
