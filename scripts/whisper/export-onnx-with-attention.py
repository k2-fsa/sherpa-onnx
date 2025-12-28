#!/usr/bin/env python3
# Copyright    2025  Posit Software, PBC
# flake8: noqa

"""
Export Whisper ONNX models with cross-attention weights for word-level timestamps.

This script exports Whisper models that include cross-attention weights from
alignment heads as an additional decoder output. These weights can be used
with Dynamic Time Warping (DTW) to compute word-level timestamps.

Based on the original export-onnx.py script.

Usage:
  python export-onnx-with-attention.py --model tiny

The exported decoder will have 4 outputs instead of 3:
  - logits
  - out_n_layer_self_k_cache
  - out_n_layer_self_v_cache
  - cross_attention_weights  (NEW: shape [n_alignment_heads, n_audio_ctx])
"""

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import onnx
import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn

import whisper
from whisper.model import (
    MultiHeadAttention,
    ResidualAttentionBlock,
    TextDecoder,
)

from export_onnx import add_meta_data, AudioEncoderTensorCache


# Alignment heads for each model variant (from whisper.cpp)
# Format: List of (layer_index, head_index) tuples
ALIGNMENT_HEADS = {
    "tiny.en": [(1, 0), (2, 0), (2, 5), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)],
    "tiny": [(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)],
    "base.en": [(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)],
    "base": [(3, 1), (4, 2), (4, 3), (4, 7), (5, 1), (5, 2), (5, 4), (5, 6)],
    "small.en": [
        (6, 6), (7, 0), (7, 3), (7, 8), (8, 2), (8, 5), (8, 7),
        (9, 0), (9, 4), (9, 8), (9, 10), (10, 0), (10, 1), (10, 2),
        (10, 3), (10, 6), (10, 11), (11, 2), (11, 4),
    ],
    "small": [
        (5, 3), (5, 9), (8, 0), (8, 4), (8, 7), (8, 8),
        (9, 0), (9, 7), (9, 9), (10, 5),
    ],
    "medium.en": [
        (11, 4), (14, 1), (14, 12), (14, 14), (15, 4), (16, 0),
        (16, 4), (16, 9), (17, 12), (17, 14), (18, 7), (18, 10),
        (18, 15), (20, 0), (20, 3), (20, 9), (20, 14), (21, 12),
    ],
    "medium": [(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)],
    "large-v1": [
        (9, 19), (11, 2), (11, 4), (11, 17), (22, 7), (22, 11),
        (22, 17), (23, 2), (23, 15),
    ],
    "large-v2": [
        (10, 12), (13, 17), (16, 11), (16, 12), (16, 13), (17, 15),
        (17, 16), (18, 4), (18, 11), (18, 19), (19, 11), (21, 2),
        (21, 3), (22, 3), (22, 9), (22, 12), (23, 5), (23, 7),
        (23, 13), (25, 5), (26, 1), (26, 12), (27, 15),
    ],
    "large-v3": [
        (7, 0), (10, 17), (12, 18), (13, 12), (16, 1), (17, 14),
        (19, 11), (21, 4), (24, 1), (25, 6),
    ],
    "large": [  # Same as large-v3
        (7, 0), (10, 17), (12, 18), (13, 12), (16, 1), (17, 14),
        (19, 11), (21, 4), (24, 1), (25, 6),
    ],
    "turbo": [(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)],
    # Distil-whisper models (alignment heads discovered empirically)
    # distil-small.en has 4 decoder layers; head (3,2) has 0.985 diagonal score
    "distil-small.en": [(3, 2)],
    # distil-medium.en has 2 decoder layers; head (1,11) has 0.804 diagonal score
    "distil-medium.en": [(1, 11)],
    # distil-large-v2 has 2 decoder layers; head (1,12) has 0.806 diagonal score
    "distil-large-v2": [(1, 12)],
    # distil-large-v3 has 2 decoder layers; head (1,3) has 0.623 diagonal score
    "distil-large-v3": [(1, 3)],
    # distil-large-v3.5 has 2 decoder layers; head (1,3) has 0.483 diagonal score
    "distil-large-v3.5": [(1, 3)],
}

# Models that require downloading from HuggingFace
DISTIL_MODELS = {
    "distil-small.en": "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/original-model.bin",
    "distil-medium.en": "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/original-model.bin",
    "distil-large-v2": "https://huggingface.co/distil-whisper/distil-large-v2/resolve/main/original-model.bin",
    "distil-large-v3": "https://huggingface.co/distil-whisper/distil-large-v3-openai/resolve/main/model.bin",
    "distil-large-v3.5": "https://huggingface.co/distil-whisper/distil-large-v3.5-openai/resolve/main/model.bin",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(ALIGNMENT_HEADS.keys()),
        help="Whisper model name (must have known alignment heads)",
    )
    return parser.parse_args()


def download_distil_model(name: str) -> str:
    """Download a distil-whisper model from HuggingFace if needed.

    Returns the path to the downloaded checkpoint file.
    """
    import urllib.request

    if name not in DISTIL_MODELS:
        raise ValueError(f"Unknown distil model: {name}")

    url = DISTIL_MODELS[name]
    # Create a local filename based on the model name
    local_filename = f"{name.replace('.', '-')}-original-model.bin"

    if os.path.exists(local_filename):
        print(f"Using existing checkpoint: {local_filename}")
        return local_filename

    print(f"Downloading {name} from {url}...")
    urllib.request.urlretrieve(url, local_filename)
    print(f"Downloaded to: {local_filename}")
    return local_filename


def load_model(name: str):
    """Load a Whisper model, handling distil models specially."""
    if name in DISTIL_MODELS:
        checkpoint_path = download_distil_model(name)
        return whisper.load_model(checkpoint_path)
    else:
        return whisper.load_model(name)


def get_alignment_heads(name: str) -> List[Tuple[int, int]]:
    """Get alignment heads for a model.

    Raises:
        ValueError: If no alignment heads are defined for the model.
    """
    if name in ALIGNMENT_HEADS:
        return ALIGNMENT_HEADS[name]

    raise ValueError(
        f"No alignment heads defined for model '{name}'. "
        f"Supported models: {', '.join(sorted(ALIGNMENT_HEADS.keys()))}"
    )


def convert_tokens(name: str, model):
    """Convert and save tokens file."""
    whisper_dir = Path(whisper.__file__).parent
    multilingual = model.is_multilingual
    tokenizer = (
        whisper_dir
        / "assets"
        / (multilingual and "multilingual.tiktoken" or "gpt2.tiktoken")
    )
    if not tokenizer.is_file():
        raise ValueError(f"Cannot find {tokenizer}")

    with open(tokenizer, "r") as f:
        contents = f.read()
        tokens = {
            token: int(rank)
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }

    output_path = f"{name}-tokens.txt"
    with open(output_path, "w") as f:
        for t, i in tokens.items():
            f.write(f"{t} {i}\n")


# =============================================================================
# Attention-enabled decoder classes
# =============================================================================


class MultiHeadAttentionCrossWithWeights(nn.Module):
    """Cross-attention that returns both output and attention weights."""

    def __init__(
        self,
        inMultiHeadAttention: MultiHeadAttention,
        layer_index: int,
        alignment_heads: List[Tuple[int, int]],
    ):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention
        self.layer_index = layer_index
        # Find which heads in this layer are alignment heads
        self.alignment_head_indices = [
            head_idx for (layer_idx, head_idx) in alignment_heads
            if layer_idx == layer_index
        ]
        self.n_head = inMultiHeadAttention.n_head

    def forward(
        self,
        x: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        q = self.multiHeadAttention.query(x)

        # Compute attention weights manually (don't use SDPA)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k_reshaped = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v_reshaped = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        # Compute QK^T with scaling
        qk = (q * scale) @ (k_reshaped * scale).transpose(-1, -2)
        qk = qk.float()

        # Softmax to get attention weights
        w = F.softmax(qk, dim=-1).to(q.dtype)

        # Compute output
        out = (w @ v_reshaped).permute(0, 2, 1, 3).flatten(start_dim=2)
        out = self.multiHeadAttention.out(out)

        # Extract alignment head weights if this layer has any
        if self.alignment_head_indices:
            # w shape: (batch, n_head, n_ctx, n_audio_ctx)
            # Select only the alignment heads for this layer
            # Output shape: (batch, n_alignment_heads, n_ctx, n_audio_ctx)
            alignment_weights = w[:, self.alignment_head_indices, :, :]
        else:
            alignment_weights = None

        return out, alignment_weights


class MultiHeadAttentionSelfManual(nn.Module):
    """Self-attention with KV cache support and manual attention computation."""

    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention
        self.n_head = inMultiHeadAttention.n_head

    def forward(
        self,
        x: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        mask: Tensor,
    ):
        q = self.multiHeadAttention.query(x)
        k = self.multiHeadAttention.key(x)
        v = self.multiHeadAttention.value(x)

        k_cache[:, -k.shape[1] :, :] = k
        v_cache[:, -v.shape[1] :, :] = v

        # Manual attention computation (avoid SDPA for ONNX compatibility)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k_reshaped = k_cache.view(*k_cache.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v_reshaped = v_cache.view(*v_cache.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = (q * scale) @ (k_reshaped * scale).transpose(-1, -2)
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        out = (w @ v_reshaped).permute(0, 2, 1, 3).flatten(start_dim=2)

        return self.multiHeadAttention.out(out), k_cache, v_cache


class ResidualAttentionBlockWithWeights(nn.Module):
    """Residual attention block that returns cross-attention weights."""

    def __init__(
        self,
        inResidualAttentionBlock: ResidualAttentionBlock,
        layer_index: int,
        alignment_heads: List[Tuple[int, int]],
    ):
        super().__init__()
        self.originalBlock = inResidualAttentionBlock
        self.attn = MultiHeadAttentionSelfManual(inResidualAttentionBlock.attn)
        self.cross_attn = (
            MultiHeadAttentionCrossWithWeights(
                inResidualAttentionBlock.cross_attn,
                layer_index,
                alignment_heads,
            )
            if inResidualAttentionBlock.cross_attn
            else None
        )

    def forward(
        self,
        x: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.attn(
            self.originalBlock.attn_ln(x), self_k_cache, self_v_cache, mask=mask
        )
        x = x + self_attn_x

        cross_attention_weights = None
        if self.cross_attn:
            cross_out, cross_attention_weights = self.cross_attn(
                self.originalBlock.cross_attn_ln(x), cross_k, cross_v
            )
            x = x + cross_out

        x = x + self.originalBlock.mlp(self.originalBlock.mlp_ln(x))
        return x, self_k_cache_updated, self_v_cache_updated, cross_attention_weights


class TextDecoderWithAttention(nn.Module):
    """Text decoder that outputs cross-attention weights from alignment heads."""

    def __init__(
        self,
        inTextDecoder: TextDecoder,
        in_n_ctx: int,
        alignment_heads: List[Tuple[int, int]],
    ):
        super().__init__()
        self.textDecoder = inTextDecoder
        self.n_ctx = in_n_ctx
        self.alignment_heads = alignment_heads

        self.blocks = nn.ModuleList()
        for i, original_block in enumerate(self.textDecoder.blocks):
            self.blocks.append(
                ResidualAttentionBlockWithWeights(original_block, i, alignment_heads)
            )

    def forward(
        self,
        tokens: Tensor,
        n_layer_self_k_cache: Tensor,
        n_layer_self_v_cache: Tensor,
        n_layer_cross_k: Tensor,
        n_layer_cross_v: Tensor,
        offset: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = (
            self.textDecoder.token_embedding(tokens)
            + self.textDecoder.positional_embedding[
                offset[0] : offset[0] + tokens.shape[-1]
            ]
        )
        x = x.to(n_layer_cross_k[0].dtype)

        # Collect attention weights from alignment heads across all layers
        all_attention_weights = []

        for i, block in enumerate(self.blocks):
            self_k_cache = n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :]
            self_v_cache = n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :]

            x, self_k_cache, self_v_cache, attn_weights = block(
                x,
                self_k_cache=self_k_cache,
                self_v_cache=self_v_cache,
                cross_k=n_layer_cross_k[i],
                cross_v=n_layer_cross_v[i],
                mask=self.textDecoder.mask,
            )

            n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_k_cache
            n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_v_cache

            if attn_weights is not None:
                all_attention_weights.append(attn_weights)

        x = self.textDecoder.ln(x)

        logits = (
            torch.matmul(
                self.textDecoder.token_embedding.weight.to(x.dtype),
                x.permute(0, 2, 1),
            )
            .permute(0, 2, 1)
            .float()
        )

        # Stack attention weights from all alignment heads
        # Shape: (batch, total_alignment_heads, n_tokens, n_audio_ctx)
        if all_attention_weights:
            cross_attention_weights = torch.cat(all_attention_weights, dim=1)
        else:
            # Fallback: create dummy tensor if no alignment heads configured
            cross_attention_weights = torch.zeros(
                tokens.shape[0], 1, tokens.shape[1], n_layer_cross_k.shape[2],
                device=tokens.device, dtype=logits.dtype
            )

        return logits, n_layer_self_k_cache, n_layer_self_v_cache, cross_attention_weights


# =============================================================================
# Main export function
# =============================================================================


@torch.no_grad()
def main():
    args = get_args()
    name = args.model

    print(f"Exporting {name} with cross-attention weights")

    opset_version = 13

    # Load model
    model = load_model(name)
    print(f"Model dimensions: {model.dims}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get alignment heads for this model
    alignment_heads = get_alignment_heads(name)
    print(f"Using {len(alignment_heads)} alignment heads: {alignment_heads}")

    convert_tokens(name=name, model=model)

    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )

    model.eval()

    # Prepare test input
    audio = torch.rand(16000 * 2)
    audio = whisper.pad_or_trim(audio)

    if name in ("distil-large-v3", "distil-large-v3.5"):
        n_mels = 128
    elif name in ("large", "large-v3", "turbo"):
        n_mels = 128
    else:
        n_mels = 80

    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(model.device).unsqueeze(0)
    batch_size = 1

    # Export encoder (same as original)
    encoder = AudioEncoderTensorCache(model.encoder, model.decoder)
    n_layer_cross_k, n_layer_cross_v = encoder(mel)

    encoder_filename = f"{name}-encoder.onnx"
    torch.onnx.export(
        encoder,
        mel,
        encoder_filename,
        opset_version=opset_version,
        input_names=["mel"],
        output_names=["n_layer_cross_k", "n_layer_cross_v"],
        dynamic_axes={
            "mel": {0: "n_audio", 2: "T"},
            "n_layer_cross_k": {1: "n_audio", 2: "T"},
            "n_layer_cross_v": {1: "n_audio", 2: "T"},
        },
    )

    encoder_meta_data = {
        "model_type": f"whisper-{name}",
        "version": "2",  # Version 2 indicates attention-enabled
        "maintainer": "k2-fsa",
        "n_mels": model.dims.n_mels,
        "n_audio_ctx": model.dims.n_audio_ctx,
        "n_audio_state": model.dims.n_audio_state,
        "n_audio_head": model.dims.n_audio_head,
        "n_audio_layer": model.dims.n_audio_layer,
        "n_vocab": model.dims.n_vocab,
        "n_text_ctx": model.dims.n_text_ctx,
        "n_text_state": model.dims.n_text_state,
        "n_text_head": model.dims.n_text_head,
        "n_text_layer": model.dims.n_text_layer,
        "sot_sequence": ",".join(list(map(str, tokenizer.sot_sequence))),
        "all_language_tokens": ",".join(list(map(str, tokenizer.all_language_tokens))),
        "all_language_codes": ",".join(tokenizer.all_language_codes),
        "sot": tokenizer.sot,
        "sot_index": tokenizer.sot_sequence.index(tokenizer.sot),
        "eot": tokenizer.eot,
        "blank_id": tokenizer.encode(" ")[0],
        "is_multilingual": int(model.is_multilingual),
        "no_speech": tokenizer.no_speech,
        "non_speech_tokens": ",".join(list(map(str, tokenizer.non_speech_tokens))),
        "transcribe": tokenizer.transcribe,
        "translate": tokenizer.translate,
        "sot_prev": tokenizer.sot_prev,
        "sot_lm": tokenizer.sot_lm,
        "no_timestamps": tokenizer.no_timestamps,
        # Attention-specific metadata
        "n_alignment_heads": len(alignment_heads),
        "alignment_heads": ",".join([f"{l}:{h}" for l, h in alignment_heads]),
    }
    print(f"Encoder metadata: {encoder_meta_data}")
    add_meta_data(filename=encoder_filename, meta_data=encoder_meta_data)

    # Export decoder with attention outputs
    n_audio = mel.shape[0]
    tokens = torch.tensor(
        [[tokenizer.sot, tokenizer.sot, tokenizer.sot]] * n_audio
    ).to(mel.device)

    decoder = TextDecoderWithAttention(
        model.decoder, model.dims.n_text_ctx, alignment_heads
    )

    n_layer_self_k_cache = torch.zeros(
        (
            len(model.decoder.blocks),
            n_audio,
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    n_layer_self_v_cache = torch.zeros(
        (
            len(model.decoder.blocks),
            n_audio,
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    offset = torch.zeros(1, dtype=torch.int64).to(mel.device)

    # Test forward pass
    logits, _, _, cross_attn_weights = decoder(
        tokens,
        n_layer_self_k_cache.clone(),
        n_layer_self_v_cache.clone(),
        n_layer_cross_k,
        n_layer_cross_v,
        offset,
    )

    print(f"Logits shape: {logits.shape}")
    print(f"Cross-attention weights shape: {cross_attn_weights.shape}")
    assert cross_attn_weights.shape == (
        n_audio, len(alignment_heads), tokens.shape[1], model.dims.n_audio_ctx
    ), f"Unexpected attention shape: {cross_attn_weights.shape}"

    # Export with single token input (for autoregressive decoding)
    offset = torch.tensor([tokens.shape[1]], dtype=torch.int64).to(mel.device)
    tokens_single = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)

    decoder_filename = f"{name}-decoder.onnx"
    torch.onnx.export(
        decoder,
        (
            tokens_single,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            offset,
        ),
        decoder_filename,
        opset_version=opset_version,
        input_names=[
            "tokens",
            "in_n_layer_self_k_cache",
            "in_n_layer_self_v_cache",
            "n_layer_cross_k",
            "n_layer_cross_v",
            "offset",
        ],
        output_names=[
            "logits",
            "out_n_layer_self_k_cache",
            "out_n_layer_self_v_cache",
            "cross_attention_weights",
        ],
        dynamic_axes={
            "tokens": {0: "n_audio", 1: "n_tokens"},
            "in_n_layer_self_k_cache": {1: "n_audio"},
            "in_n_layer_self_v_cache": {1: "n_audio"},
            "n_layer_cross_k": {1: "n_audio", 2: "T"},
            "n_layer_cross_v": {1: "n_audio", 2: "T"},
            "cross_attention_weights": {0: "n_audio", 2: "n_tokens", 3: "T"},
        },
    )

    if "large" in name:
        decoder_external_filename = decoder_filename.split(".onnx")[0]
        decoder_model = onnx.load(decoder_filename)
        onnx.save(
            decoder_model,
            decoder_filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=decoder_external_filename + ".weights",
        )

    # Generate int8 quantized models
    print("Generating int8 quantized models...")

    encoder_filename_int8 = f"{name}-encoder.int8.onnx"
    quantize_dynamic(
        model_input=encoder_filename,
        model_output=encoder_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    decoder_filename_int8 = f"{name}-decoder.int8.onnx"
    quantize_dynamic(
        model_input=decoder_filename,
        model_output=decoder_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    print(f"\nExported files:")
    print(f"  - {encoder_filename}")
    print(f"  - {encoder_filename_int8}")
    print(f"  - {decoder_filename}")
    print(f"  - {decoder_filename_int8}")
    print(f"  - {name}-tokens.txt")
    print(f"\nDecoder has 4 outputs including cross_attention_weights")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # To fix
    # TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not Tensor
    # See also https://github.com/k2-fsa/sherpa-onnx/issues/1764
    from whisper.model import disable_sdpa

    with disable_sdpa():
        main()
