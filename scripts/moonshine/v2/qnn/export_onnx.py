#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
Export moonshine to ONNX for QNN (all fixed shapes).

Encoder: computes cross-attention K/V caches.
Decoder: computes delta self-attention K/V using split attention pattern.
         Cache updated externally by the caller.

Usage:
  cd scripts/moonshine/v2/qnn
  python3 export_onnx.py --max-len 10
"""

import argparse
import base64
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoModelForSpeechSeq2Seq


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, filename)


def causal_mask_1d(n: int, L: int, device=None, dtype=torch.int32):
    """1-D mask: 0=allowed, 1=masked. Positions [:n] are allowed."""
    mask = torch.ones((L,), device=device, dtype=dtype)
    if n > 0:
        mask[:n] = 0
    return mask


def patch_layernorm_bias(model):
    """Add dummy bias to LayerNorm layers that don't have bias (for QNN compatibility)."""
    for module in model.modules():
        if isinstance(module, nn.LayerNorm) and module.bias is None:
            module.bias = nn.Parameter(torch.zeros(module.normalized_shape, device=module.weight.device, dtype=module.weight.dtype))
    return model



# ---------------------------------------------------------------------------
# Encoder: returns per-layer cross K/V
# ---------------------------------------------------------------------------
class AudioEncoderTensorCache(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
        self.decoder_layers = model.model.decoder.layers

    def forward(self, audio: Tensor):
        encoder_out = self.encoder(audio).last_hidden_state

        cross_kv_pair = []
        for layer in self.decoder_layers:
            k = layer.encoder_attn.k_proj(encoder_out)
            v = layer.encoder_attn.v_proj(encoder_out)
            cross_kv_pair.append((k, v))

        return cross_kv_pair


# ---------------------------------------------------------------------------
# Split self-attention: compute q@k_cache and q@k_new separately
# ---------------------------------------------------------------------------
def split_self_attention(
    q: Tensor,  # (1, 1, hidden)
    k_cache: Tensor,  # (1, max_seq, hidden)
    v_cache: Tensor,  # (1, max_seq, hidden)
    k_new: Tensor,  # (1, 1, hidden)
    v_new: Tensor,  # (1, 1, hidden)
    mask: Tensor,  # (max_seq,) int
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    head_dim_padding: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    bsz = 1
    max_seq = k_cache.shape[1]
    scale = head_dim**-0.25

    q = q.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
    k_c = k_cache.view(bsz, max_seq, num_kv_heads, head_dim).transpose(1, 2)
    v_c = v_cache.view(bsz, max_seq, num_kv_heads, head_dim).transpose(1, 2)
    k_n = k_new.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
    v_n = v_new.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)

    if head_dim_padding > 0:
        q = F.pad(q, (0, head_dim_padding))
        k_c = F.pad(k_c, (0, head_dim_padding))
        v_c = F.pad(v_c, (0, head_dim_padding))
        k_n = F.pad(k_n, (0, head_dim_padding))
        v_n = F.pad(v_n, (0, head_dim_padding))

    if num_heads != num_kv_heads:
        r = num_heads // num_kv_heads
        k_c = k_c.repeat_interleave(r, dim=1)
        v_c = v_c.repeat_interleave(r, dim=1)
        k_n = k_n.repeat_interleave(r, dim=1)
        v_n = v_n.repeat_interleave(r, dim=1)

    # Move scale after matmul to avoid intermediate tensors for QNN compatibility
    scale_sq = scale * scale
    qk_cache = (q @ k_c.transpose(-1, -2)) * scale_sq
    qk_new = (q @ k_n.transpose(-1, -2)) * scale_sq

    qk_cache = qk_cache.masked_fill(mask.view(1, 1, 1, max_seq).to(torch.bool), -60000)

    qk_total = torch.cat([qk_cache, qk_new], dim=-1)
    w_total = F.softmax(qk_total.float(), dim=-1).to(q.dtype)

    w_cache = w_total[:, :, :, :-1]
    w_new = w_total[:, :, :, -1:]

    out = w_cache @ v_c + w_new @ v_n

    if head_dim_padding > 0:
        out = out[..., :head_dim]

    out = out.permute(0, 2, 1, 3).contiguous().view(bsz, 1, -1)
    return out, k_new, v_new


# ---------------------------------------------------------------------------
# Cross attention
# ---------------------------------------------------------------------------
def cross_attention(
    q: Tensor,  # (1, 1, hidden)
    k: Tensor,  # (1, enc_seq, hidden)
    v: Tensor,  # (1, enc_seq, hidden)
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    head_dim_padding: int,
) -> Tensor:
    bsz = 1
    enc_seq = k.shape[1]
    scale = head_dim**-0.25

    q = q.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, enc_seq, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, enc_seq, num_kv_heads, head_dim).transpose(1, 2)

    if head_dim_padding > 0:
        q = F.pad(q, (0, head_dim_padding))
        k = F.pad(k, (0, head_dim_padding))
        v = F.pad(v, (0, head_dim_padding))

    if num_heads != num_kv_heads:
        r = num_heads // num_kv_heads
        q = q.repeat_interleave(r, dim=1)
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)

    # Move scale after matmul to avoid intermediate tensors for QNN compatibility
    scale_sq = scale * scale
    w = F.softmax((q @ k.transpose(-1, -2)) * scale_sq, dim=-1).to(q.dtype)
    out = w @ v

    if head_dim_padding > 0:
        out = out[..., :head_dim]

    return out.permute(0, 2, 1, 3).contiguous().view(bsz, 1, -1)


# ---------------------------------------------------------------------------
# Decoder wrapper
# ---------------------------------------------------------------------------
class TextDecoderTensorCache(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        self.num_layers = len(model.model.decoder.layers)
        self.config = model.config
        #  print("emb.config", self.decoder.rotary_emb.config)

    def forward(
        self,
        tokens: Tensor,  # (1, 1)
        self_kv_pair: List[Tuple[Tensor, Tensor]],  # [(k_cache, v_cache)] per layer
        cross_kv_pair: List[Tuple[Tensor, Tensor]],  # [(cross_k, cross_v)] per layer
        offset: Tensor,  # (1,) int32
        mask: Tensor,  # (max_seq,) int32
    ):
        device = tokens.device

        # Always decoding 1 token: position_id = offset
        position_ids = offset.to(torch.long).reshape(1, 1)  # (1, 1)
        hidden_states = self.decoder.embed_tokens(tokens.to(torch.long))
        position_embeddings = self.decoder.rotary_emb(
            hidden_states, position_ids=position_ids
        )

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)  # (1, 1, 1, rotary_dim)
        sin = sin.unsqueeze(1)
        rotary_dim = cos.shape[-1]

        # Interleave cos/sin without repeat_interleave (avoids 5D tensors for QNN)
        # cos[..., :half] has shape (1, 1, 1, half), expand to (1, 1, 1, rotary_dim)
        cos_half = cos[..., : rotary_dim // 2]
        sin_half = sin[..., : rotary_dim // 2]
        cos_expanded = torch.zeros_like(cos)
        sin_expanded = torch.zeros_like(sin)
        cos_expanded[..., 0::2] = cos_half
        cos_expanded[..., 1::2] = cos_half
        sin_expanded[..., 0::2] = sin_half
        sin_expanded[..., 1::2] = sin_half

        # Expand to match num_heads: (1, 1, 1, rotary_dim) -> (1, num_heads, 1, rotary_dim)
        num_heads = self.decoder.layers[0].self_attn.config.num_attention_heads
        cos_expanded = cos_expanded.repeat(1, num_heads, 1, 1)
        sin_expanded = sin_expanded.repeat(1, num_heads, 1, 1)

        def rotate_half(x):
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)

        def apply_rotary(x):
            x_rot = x[..., :rotary_dim]
            x_pass = x[..., rotary_dim:]
            x_rot = x_rot * cos_expanded + rotate_half(x_rot) * sin_expanded
            return torch.cat([x_rot, x_pass], dim=-1)

        this_self_kv_pair = []

        for i, layer in enumerate(self.decoder.layers):
            num_heads = layer.self_attn.config.num_attention_heads
            num_kv_heads = layer.self_attn.config.num_key_value_heads
            head_dim = layer.self_attn.head_dim
            head_dim_padding = layer.self_attn.head_dim_padding

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            q = layer.self_attn.q_proj(hidden_states)
            k_new = layer.self_attn.k_proj(hidden_states)
            v_new = layer.self_attn.v_proj(hidden_states)

            bsz = 1
            q_r = q.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
            k_r = k_new.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
            q_r = apply_rotary(q_r)
            k_r = apply_rotary(k_r)
            q = q_r.transpose(1, 2).contiguous().view(bsz, 1, -1)
            k_new = k_r.transpose(1, 2).contiguous().view(bsz, 1, -1)

            self_k_cache = self_kv_pair[i][0]
            self_v_cache = self_kv_pair[i][1]

            self_out, k_delta, v_delta = split_self_attention(
                q,
                self_k_cache,
                self_v_cache,
                k_new,
                v_new,
                mask,
                num_heads,
                num_kv_heads,
                head_dim,
                head_dim_padding,
            )

            this_self_kv_pair.append((k_delta, v_delta))
            hidden_states = residual + layer.self_attn.o_proj(self_out)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            q_cross = layer.encoder_attn.q_proj(hidden_states)
            cross_k = cross_kv_pair[i][0]
            cross_v = cross_kv_pair[i][1]

            cross_out = cross_attention(
                q_cross,
                cross_k,
                cross_v,
                num_heads,
                num_kv_heads,
                head_dim,
                head_dim_padding,
            )
            hidden_states = residual + layer.encoder_attn.o_proj(cross_out)

            residual = hidden_states
            hidden_states = layer.final_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.decoder.norm(hidden_states)
        logits = self.proj_out(hidden_states)

        return logits, this_self_kv_pair


def token_to_bytes(token: str) -> bytes:
    """Convert a token string to bytes, handling byte tokens like <0xE5>."""
    import re

    match = re.match(r"^<0x([0-9A-Fa-f]{2})>$", token)
    if match:
        return bytes([int(match.group(1), 16)])
    else:
        return token.encode("utf-8")


def generate_tokens(output_dir: str, model):
    """Generate tokens.txt from the model's tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    vocab = tokenizer.get_vocab()

    output_path = Path(output_dir) / "tokens.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            token_bytes = token_to_bytes(token)
            b64 = base64.b64encode(token_bytes).decode("ascii")
            f.write(f"{b64} {idx}\n")

    print(f"Saved {len(vocab)} tokens to {output_path}")
    return tokenizer


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="UsefulSensors/moonshine-base-zh"
    )
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument(
        "--max-len",
        type=int,
        default=10,
        help="Max audio length in seconds (default: 10)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_audio_len = args.max_len * 16000  # samples at 16kHz

    print(f"Loading model: {args.model_name}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    # Patch LayerNorm to have bias for QNN compatibility
    model = patch_layernorm_bias(model)

    config = model.config
    num_layers = len(model.model.decoder.layers)
    num_heads = config.decoder_num_attention_heads
    num_kv_heads = config.decoder_num_key_value_heads
    head_dim = config.hidden_size // num_heads
    hidden_size = config.hidden_size
    max_seq_len = config.max_position_embeddings

    print(
        f"Config: layers={num_layers}, heads={num_heads}, kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}, hidden={hidden_size}, vocab={config.vocab_size}, "
        f"max_seq={max_seq_len}, max_audio_len={max_audio_len}"
    )

    generate_tokens(str(output_dir), model)

    # Compute encoder output length by running a dummy forward pass
    dummy_audio = torch.randn(1, max_audio_len, dtype=torch.float32)
    encoder = AudioEncoderTensorCache(model)
    cross_kv_pair = encoder(dummy_audio)
    enc_seq_len = cross_kv_pair[0][0].shape[1]
    print(f"Encoder output seq_len: {enc_seq_len} (for {args.max_len}s audio)")

    meta = {
        "model_type": "moonshine",
        "version": "1",
        "maintainer": "k2-fsa",
        "hidden_size": str(hidden_size),
        "num_attention_heads": str(num_heads),
        "num_key_value_heads": str(num_kv_heads),
        "head_dim": str(head_dim),
        "num_decoder_layers": str(num_layers),
        "vocab_size": str(config.vocab_size),
        "partial_rotary_factor": str(config.partial_rotary_factor),
        "sampling_rate": "16000",
        "max_seq_len": str(max_seq_len),
        "max_audio_len": str(max_audio_len),
        "enc_seq_len": str(enc_seq_len),
    }

    # --- Export encoder (fixed shapes) ---
    print("\nExporting encoder...")
    encoder_filename = str(output_dir / "encoder.onnx")
    output_names = []
    for i in range(num_layers):
        output_names.append(f"cross_k_{i}")
        output_names.append(f"cross_v_{i}")

    torch.onnx.export(
        encoder,
        (dummy_audio,),
        encoder_filename,
        dynamo=True,
        opset_version=18,
        input_names=["audio"],
        output_names=output_names,
    )

    # Re-save without external data to avoid duplication
    model_onnx = onnx.load(encoder_filename)
    onnx.save(model_onnx, encoder_filename)
    data_file = encoder_filename + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    # Simplify with onnxsim to fix QNN converter issues
    try:
        import onnxsim
        print("  Simplifying encoder with onnxsim...")
        model_onnx = onnx.load(encoder_filename)
        model_sim, ok = onnxsim.simplify(model_onnx)
        if ok:
            onnx.save(model_sim, encoder_filename)
            print("  Simplified successfully")
        else:
            print("  Warning: onnxsim failed, using original model")
    except ImportError:
        print("  Warning: onnxsim not installed, skipping simplification")


    add_meta_data(encoder_filename, meta)
    print(f"  Saved to {encoder_filename}")

    # --- Export decoder (fixed shapes) ---
    print("\nExporting decoder...")
    decoder = TextDecoderTensorCache(model)

    dummy_tokens = torch.tensor([[1]], dtype=torch.int32)
    dummy_self_kv = []
    for _ in range(num_layers):
        dummy_self_kv.append(
            (
                torch.zeros(1, max_seq_len, hidden_size),
                torch.zeros(1, max_seq_len, hidden_size),
            )
        )
    dummy_cross_kv = []
    for _ in range(num_layers):
        dummy_cross_kv.append(
            (
                torch.zeros(1, enc_seq_len, hidden_size),
                torch.zeros(1, enc_seq_len, hidden_size),
            )
        )
    dummy_offset = torch.tensor([0], dtype=torch.int32)
    dummy_mask = causal_mask_1d(0, max_seq_len)

    logits, this_kv = decoder(
        dummy_tokens, dummy_self_kv, dummy_cross_kv, dummy_offset, dummy_mask
    )
    print(f"  logits: {logits.shape}, delta_k: {this_kv[0][0].shape}")

    input_names = ["tokens"]
    for i in range(num_layers):
        input_names.append(f"self_k_{i}")
        input_names.append(f"self_v_{i}")
    for i in range(num_layers):
        input_names.append(f"cross_k_{i}")
        input_names.append(f"cross_v_{i}")
    input_names.append("offset")
    input_names.append("mask")

    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"this_self_k_{i}")
        output_names.append(f"this_self_v_{i}")

    decoder_filename = str(output_dir / "decoder.onnx")
    torch.onnx.export(
        decoder,
        (dummy_tokens, dummy_self_kv, dummy_cross_kv, dummy_offset, dummy_mask),
        decoder_filename,
        dynamo=True,
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
    )

    # Re-save without external data to avoid duplication
    model_onnx = onnx.load(decoder_filename)
    onnx.save(model_onnx, decoder_filename)
    data_file = decoder_filename + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    # Simplify with onnxsim to fix QNN converter issues
    try:
        import onnxsim
        print("  Simplifying decoder with onnxsim...")
        model_onnx = onnx.load(decoder_filename)
        model_sim, ok = onnxsim.simplify(model_onnx)
        if ok:
            onnx.save(model_sim, decoder_filename)
            print("  Simplified successfully")
        else:
            print("  Warning: onnxsim failed, using original model")
    except ImportError:
        print("  Warning: onnxsim not installed, skipping simplification")


    add_meta_data(decoder_filename, meta)
    print(f"  Saved to {decoder_filename}")

    print(f"\nDone! Exported:")
    print(f"  {encoder_filename}")
    print(f"  {decoder_filename}")
    print(f"  {output_dir / 'tokens.txt'}")


if __name__ == "__main__":
    main()
