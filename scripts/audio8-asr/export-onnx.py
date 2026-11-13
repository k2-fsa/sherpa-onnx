"""Export Audio8-ASR-0.1B to ONNX for a sherpa-onnx style C++ runtime.

Two graphs:

  encoder.onnx
      inputs : audios float32 [1, 128, 3000]   (Whisper log-mel, zero padded)
               audio_feature_lengths int64 [1]  (ceil(samples / 160))
      outputs: hidden float32 [1, 390, 1024]    (MLP-tower output, rows after
                                                 the valid length are zeros)
               valid_mask float32 [1, 390]       (1 for valid rows)

  decoder.onnx
      inputs : input_ids int64 [1, T]
               audio_hidden float32 [1, K, 1024] (gathered + pooled encoder rows)
               attention_mask int64 [1, 512]
               cache_position int64 [T]
               cache_key_i / cache_value_i float32 [1, 8, 512, 64], i = 0..7
      outputs: logits float32 [1, T, 151936]
               present_key_i / present_value_i float32 [1, 8, T, 64]

The decoder applies the audio projector (LayerNorm + Linear 1024 -> 512), injects
the projected rows at the `<|audio|>` token positions and runs the 8-layer Qwen2
language model with a static 512-slot KV cache.

Usage:
    python export_audio8_onnx.py --out-dir ../exported
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent


def add_meta_data(filename, meta_data):
    """Add meta data to an ONNX model. It is changed in-place.

    Supports models whose weights live in an external .onnx.data file.
    """
    import onnx

    data_file = Path(str(filename) + ".data") if str(filename).endswith(".onnx") else Path(str(filename) + ".data")
    # do not drop the weights when they live in an external data file
    model = onnx.load(str(filename), load_external_data=data_file.exists())

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    if data_file.exists():
        onnx.save(model, str(filename),
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=data_file.name,
                  size_threshold=1024)
    else:
        onnx.save(model, str(filename))

MODEL_DIR = os.environ.get("AUDIO8_HF_DIR", str(ROOT / "Audio8-ASR-0.1B"))

FRAMES = 3000          # 30 s of 10 ms mel frames
ENC_TOKENS = 390       # encoder output rows for 3000 frames (13 per 100 frames)
MAX_TOTAL_LEN = 512
AUDIO_TOKEN_ID = 151646



def restore_sinusoid_positions(model):
    """transformers 5.22.2 zero-fills the non-persistent sinusoid buffer.

    `SinusoidsPositionEmbedding` is not stored in the checkpoint, so the loader
    has to recompute it (the checkpoint was written with transformers 4.57.3).
    Without this the audio encoder silently loses positional information.
    """
    import numpy as np
    pe = model.audio_encoder.positional_embedding.positional_embedding
    ch = pe.shape[1]
    inc = np.log(10000) / (ch // 2 - 1)
    inv = np.exp(-inc * np.arange(ch // 2))
    scaled = np.arange(pe.shape[0])[:, None] * inv[None, :]
    table = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    with torch.no_grad():
        pe.copy_(torch.from_numpy(table).to(pe.dtype))
    return float(pe.abs().max())

# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
class _AttnExport(nn.Module):
    """Audio self-attention with an additive mask (block windows + valid rows)."""

    def __init__(self, attn, n_heads: int):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (
            attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj)
        self.n_heads = n_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.scaling

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.n_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(n, self.n_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(n, self.n_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scaling + bias
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v).transpose(0, 1).reshape(n, -1)
        return self.out_proj(out)


class _LayerExport(nn.Module):
    def __init__(self, layer, n_heads: int):
        super().__init__()
        self.attn = _AttnExport(layer.self_attn, n_heads)
        self.attn_ln = layer.self_attn_layer_norm
        self.ffn_ln = layer.final_layer_norm
        self.fc1, self.fc2, self.act = layer.fc1, layer.fc2, layer.activation_fn

    def forward(self, h: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.attn_ln(h), bias)
        x = self.fc2(self.act(self.fc1(self.ffn_ln(h))))
        return h + x


class EncoderExport(nn.Module):
    """Audio encoder + MLP tower with a static 30 s window.

    The HF encoder uses ``pad_sequence`` plus a data dependent ``split`` on the mel
    frames, neither of which can be traced. With a fixed number of 100-frame
    chunks the conv part is static; the attention windows are `n_window_infer`
    (8 chunks = 104 tokens) and are masked by `num_rows`, which the runtime sets
    to the number of rows that the real audio length produces. That reproduces
    the variable-length HF graph exactly.
    """

    CHUNK = 100
    TOKENS_PER_CHUNK = 13
    WINDOW = 104                 # 13 * (n_window_infer // (n_window * 2))
    N_CHUNKS = FRAMES // CHUNK                    # 30
    N_TOKENS = N_CHUNKS * TOKENS_PER_CHUNK        # 390

    def __init__(self, model):
        super().__init__()
        enc = model.audio_encoder
        self.conv1, self.conv2, self.conv3 = enc.conv2d1, enc.conv2d2, enc.conv2d3
        self.conv_out = enc.conv_out
        self.layers = nn.ModuleList(
            [_LayerExport(layer, int(enc.config.encoder_attention_heads)) for layer in enc.layers])
        self.ln_post = enc.ln_post
        self.proj1, self.act, self.proj2 = enc.proj1, enc.act, enc.proj2
        self.tower = model.audio_mlp_tower
        self.register_buffer(
            "pos", enc.positional_embedding.positional_embedding[: self.TOKENS_PER_CHUNK].clone())
        idx = torch.arange(self.N_TOKENS)
        self.register_buffer("same_block", (idx[:, None] // self.WINDOW) == (idx[None, :] // self.WINDOW))
        self.register_buffer("rows", idx)

    def forward(self, audios: torch.Tensor, num_rows: torch.Tensor):
        x = audios[0].transpose(0, 1).reshape(self.N_CHUNKS, self.CHUNK, 128)
        x = x.transpose(1, 2).unsqueeze(1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.permute(0, 3, 1, 2).reshape(self.N_CHUNKS, self.TOKENS_PER_CHUNK, -1)
        x = self.conv_out(x) + self.pos.unsqueeze(0)
        h = x.reshape(self.N_TOKENS, -1)

        valid = self.rows < num_rows.reshape(())                      # [390]
        allowed = self.same_block & valid[:, None] & valid[None, :]   # [390,390]
        bias = torch.where(allowed, torch.zeros_like(h[:1, :1]),
                           torch.full_like(h[:1, :1], -1e4))          # [390,390]

        for layer in self.layers:
            h = layer(h, bias)
        h = self.ln_post(h)
        h = self.act(self.proj1(h))
        h = self.proj2(h)
        return self.tower(h).unsqueeze(0)


# --------------------------------------------------------------------------- #
# decoder (manual Qwen2 with a static KV cache)
# --------------------------------------------------------------------------- #
class DecoderExport(nn.Module):
    def __init__(self, model, max_total_len: int = MAX_TOTAL_LEN):
        super().__init__()
        lm = model.language_model
        cfg = lm.config
        self.embed = lm.model.embed_tokens
        self.layers = lm.model.layers
        self.norm = lm.model.norm
        self.lm_head = lm.lm_head
        self.projector_norm = model.audio_projector[0]
        self.projector_linear = model.audio_projector[1]
        self.num_layers = len(self.layers)
        self.num_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // self.num_heads))
        self.hidden_size = int(cfg.hidden_size)
        self.max_total_len = max_total_len
        rope_params = getattr(cfg, "rope_parameters", None) or {}
        self.rope_theta = float(rope_params.get("rope_theta", getattr(cfg, "rope_theta", 10000.0)))
        self.eps = float(getattr(cfg, "rms_norm_eps", 1e-6))

        inv = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    # rotary -----------------------------------------------------------------
    def _rope(self, x: torch.Tensor, positions: torch.Tensor):
        # x: [1, H, T, D]
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]   # [T, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)                                 # [T, D]
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        def rotate_half(t):
            t1, t2 = t[..., : self.head_dim // 2], t[..., self.head_dim // 2:]
            return torch.cat([-t2, t1], dim=-1)

        return x * cos + rotate_half(x) * sin

    def forward(self, input_ids, audio_hidden, attention_mask, cache_position, *caches):
        # caches: key0, value0, key1, value1, ...
        cache_position = cache_position.reshape(-1)
        b, t = input_ids.shape
        embeds = self.embed(input_ids)                                   # [1, T, 512]

        projected = self.projector_linear(self.projector_norm(audio_hidden))   # [1, K, 512]
        # Inject the audio rows at the `<|audio|>` positions without Scatter:
        # for every audio token (they are contiguous, but this works either way)
        # gather projected[cumsum(mask) - 1] and select it with where().
        mask = (input_ids == AUDIO_TOKEN_ID)                                # [1, T]
        idx = torch.cumsum(mask.to(torch.int64), dim=1) - 1                 # [1, T]
        k = audio_hidden.shape[1]
        idx = torch.minimum(torch.maximum(idx, torch.zeros_like(idx)), idx.new_tensor(k - 1))
        flat = idx[0][:, None].expand(-1, self.hidden_size)                 # [T, 512]
        gathered = projected[0].gather(0, flat).unsqueeze(0)                # [1, T, 512]
        embeds = torch.where(mask.unsqueeze(-1), gathered, embeds)

        # additive mask over the 512 cache slots: allowed where attention_mask==1
        kv_mask = attention_mask[0].to(torch.float32)                    # [512]
        # causal within the current chunk (positions are consecutive)
        q_pos = cache_position[:, None].to(torch.float32)                # [T,1]
        k_pos = torch.arange(self.max_total_len, device=input_ids.device).to(torch.float32)
        causal = (k_pos[None, :] <= q_pos).to(torch.float32)             # [T,512]
        slot_mask = causal * kv_mask[None, :]
        add_mask = (1.0 - slot_mask)[None, None, :, :] * torch.finfo(torch.float32).min

        h = embeds
        onehot = F.one_hot(cache_position.long(), num_classes=self.max_total_len).to(h.dtype)
        presents = []
        for i, layer in enumerate(self.layers):
            residual = h
            x = layer.input_layernorm(h)
            attn = layer.self_attn
            q = attn.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = self._rope(q, cache_position), self._rope(k, cache_position)

            past_k, past_v = caches[2 * i], caches[2 * i + 1]            # [1,Hv,512,D]
            # write the new keys/values into a copy of the cache
            k_full = past_k + torch.einsum("qs,bhqd->bhsd", onehot, k)
            v_full = past_v + torch.einsum("qs,bhqd->bhsd", onehot, v)
            presents.extend([k, v])

            attn_scores = torch.matmul(q, k_full.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_scores = attn_scores + add_mask[:, :, :, :]
            probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(probs, v_full)
            out = out.transpose(1, 2).reshape(b, t, self.hidden_size)
            h = residual + attn.o_proj(out)

            residual = h
            x = layer.post_attention_layernorm(h)
            h = residual + layer.mlp(x)

        h = self.norm(h)
        logits = self.lm_head(h)
        return (logits, *presents)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "exported"))
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="Local directory with the HF checkpoint and tokenizer")
    ap.add_argument("--max-total-len", type=int, default=MAX_TOTAL_LEN)
    ap.add_argument("--int8", action="store_true",
                    help="Also write dynamic-int8 copies (encoder.int8.onnx, "
                         "decoder.int8.onnx)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="eager").eval()
    lm = model.language_model
    lm.lm_head.weight = lm.model.embed_tokens.weight
    pos_max = restore_sinusoid_positions(model)
    print(f"restored sinusoid positions, max={pos_max:.4f}", flush=True)
    torch.set_grad_enabled(False)

    # ---------------- encoder
    enc = EncoderExport(model).eval()
    audios = torch.zeros(1, 128, FRAMES)
    num_rows = torch.tensor([390], dtype=torch.int64)
    torch.onnx.export(
        enc, (audios, num_rows), str(out / "encoder.onnx"),
        input_names=["audios", "num_rows"], output_names=["hidden"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print("encoder.onnx written", flush=True)

    # ---------------- decoder
    dec = DecoderExport(model, args.max_total_len).eval()
    t = 8
    input_ids = torch.zeros(1, t, dtype=torch.int64)
    audio_hidden = torch.zeros(1, 4, 1024)

    attn_mask = torch.zeros(1, args.max_total_len, dtype=torch.int64)
    attn_mask[:, :t] = 1
    cache_pos = torch.arange(t, dtype=torch.int64)
    caches = []
    for _ in range(dec.num_layers):
        caches.append(torch.zeros(1, dec.num_kv_heads, args.max_total_len, dec.head_dim))
        caches.append(torch.zeros(1, dec.num_kv_heads, args.max_total_len, dec.head_dim))

    names_in = ["input_ids", "audio_hidden", "attention_mask", "cache_position"]
    names_out = ["logits"]
    for i in range(dec.num_layers):
        names_in += [f"cache_key_{i}", f"cache_value_{i}"]
        names_out += [f"present_key_{i}", f"present_value_{i}"]

    torch.onnx.export(
        dec, (input_ids, audio_hidden, attn_mask, cache_pos, *caches),
        str(out / "decoder.onnx"),
        input_names=names_in, output_names=names_out,
        opset_version=17, do_constant_folding=True, dynamo=False,
        dynamic_axes={"input_ids": {1: "seq_len"}, "audio_hidden": {1: "num_audio"},
                      "cache_position": {0: "seq_len"},
                      "logits": {1: "seq_len"},
                      **{f"present_key_{i}": {2: "seq_len"} for i in range(dec.num_layers)},
                      **{f"present_value_{i}": {2: "seq_len"} for i in range(dec.num_layers)}},
    )
    print("decoder.onnx written", flush=True)

    encoder_meta = {
        "model_type": "audio8-asr",
        "version": "1",
        "maintainer": "k2-fsa",
        "url": "https://huggingface.co/Edge0/Audio8-ASR-0.1B",
        "license": "Creative Commons Attribution Non Commercial 4.0",
        "sample_rate": 16000,
        "feature_dim": 128,
        "frames": FRAMES,
        "encoder_tokens": ENC_TOKENS,
        "encoder_hidden_size": 1024,
        "audio_token_id": AUDIO_TOKEN_ID,
    }
    decoder_meta = {
        "model_type": "audio8-asr",
        "version": "1",
        "maintainer": "k2-fsa",
        "url": "https://huggingface.co/Edge0/Audio8-ASR-0.1B",
        "license": "Creative Commons Attribution Non Commercial 4.0",
        "num_layers": dec.num_layers,
        "num_attention_heads": dec.num_heads,
        "num_key_value_heads": dec.num_kv_heads,
        "head_dim": dec.head_dim,
        "hidden_size": dec.hidden_size,
        "max_total_len": args.max_total_len,
        "audio_token_id": AUDIO_TOKEN_ID,
        "vocab_size": int(lm.config.vocab_size),
    }
    add_meta_data(out / "encoder.onnx", encoder_meta)
    add_meta_data(out / "decoder.onnx", decoder_meta)
    print("meta data added to encoder.onnx / decoder.onnx", flush=True)

    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        for name in ("encoder", "decoder"):
            src = out / f"{name}.onnx"
            dst = out / f"{name}.int8.onnx"
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8,
                             per_channel=True,
                             extra_options={"EnableSubgraph": False})
            add_meta_data(dst, encoder_meta if name == "encoder" else decoder_meta)
            print(f"{name}.int8.onnx written", flush=True)

    meta = {
        "frames": FRAMES, "enc_tokens": ENC_TOKENS,
        "max_total_len": args.max_total_len,
        "num_layers": dec.num_layers, "num_heads": dec.num_heads,
        "num_kv_heads": dec.num_kv_heads, "head_dim": dec.head_dim,
        "hidden_size": dec.hidden_size, "audio_token_id": AUDIO_TOKEN_ID,
        "vocab_size": int(lm.config.vocab_size),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    print("metadata.json:", json.dumps(meta))


if __name__ == "__main__":
    main()
