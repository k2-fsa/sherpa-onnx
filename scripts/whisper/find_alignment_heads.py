#!/usr/bin/env python3
"""
Find alignment heads for a Whisper model by analyzing cross-attention patterns.

Alignment heads are attention heads that show monotonically increasing patterns,
meaning they attend to progressively later parts of the audio as more text tokens
are decoded. These heads are useful for computing word-level timestamps.

Usage:
    python find_alignment_heads.py --model distil-small.en --audio 0.wav
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import whisper
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim


def get_args():
    parser = argparse.ArgumentParser(description="Find alignment heads in Whisper models")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., distil-small.en)")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top heads to report")
    return parser.parse_args()


class AttentionCaptureHook:
    """Hook to capture cross-attention weights from all layers/heads."""

    def __init__(self):
        self.attention_weights: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.handles = []

    def hook_fn(self, layer_idx: int):
        def fn(module, input, output):
            # output is (attn_output, attn_weights) when output_attentions=True
            # But whisper doesn't use output_attentions, so we need to compute manually
            pass
        return fn

    def clear(self):
        self.attention_weights.clear()


def compute_cross_attention_weights(
    model: whisper.Whisper,
    audio_path: str,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[int], str]:
    """
    Run transcription and capture cross-attention weights from all heads.

    Returns:
        attention_weights: Dict mapping (layer, head) to attention matrix [n_tokens, n_audio_frames]
        token_ids: List of decoded token IDs
        text: Transcribed text
    """
    # Load and preprocess audio
    audio = load_audio(audio_path)
    audio = pad_or_trim(audio)

    n_mels = model.dims.n_mels
    mel = log_mel_spectrogram(audio, n_mels=n_mels).to(model.device)

    # Encode audio
    audio_features = model.encoder(mel.unsqueeze(0))

    # Get tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual,
        num_languages=getattr(model, 'num_languages', None) or (99 if model.is_multilingual else None),
        task="transcribe",
    )

    # Initial tokens (SOT sequence)
    tokens = list(tokenizer.sot_sequence)

    # Storage for attention weights per (layer, head)
    all_attention_weights: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)

    n_layers = len(model.decoder.blocks)
    n_heads = model.dims.n_text_head

    print(f"Model has {n_layers} decoder layers with {n_heads} attention heads each")

    # Decode with attention capture
    max_tokens = 448  # max context length

    for i in range(max_tokens):
        tokens_tensor = torch.tensor([tokens]).to(model.device)

        # We need to manually run through decoder blocks to capture attention
        x = model.decoder.token_embedding(tokens_tensor) + model.decoder.positional_embedding[:tokens_tensor.shape[1]]
        x = x.to(audio_features.dtype)

        for layer_idx, block in enumerate(model.decoder.blocks):
            # Self-attention (we don't need this for alignment)
            x = x + block.attn(block.attn_ln(x), mask=model.decoder.mask)[0]

            # Cross-attention - compute manually to get weights
            cross_attn = block.cross_attn
            ln_output = block.cross_attn_ln(x)

            q = cross_attn.query(ln_output)
            k = cross_attn.key(audio_features)
            v = cross_attn.value(audio_features)

            # Reshape for multi-head attention
            batch_size, n_ctx, n_state = q.shape
            head_dim = n_state // n_heads

            q = q.view(batch_size, n_ctx, n_heads, head_dim).permute(0, 2, 1, 3)
            k = k.view(batch_size, -1, n_heads, head_dim).permute(0, 2, 1, 3)
            v = v.view(batch_size, -1, n_heads, head_dim).permute(0, 2, 1, 3)

            # Compute attention weights
            scale = head_dim ** -0.25
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            attn_weights = torch.softmax(qk.float(), dim=-1)  # [batch, heads, n_ctx, n_audio]

            # Store attention weights for each head (only the last token's attention)
            for head_idx in range(n_heads):
                # Get attention from the last decoded token
                weights = attn_weights[0, head_idx, -1, :].detach().cpu().numpy()
                all_attention_weights[(layer_idx, head_idx)].append(weights)

            # Compute attention output
            attn_output = (attn_weights.to(v.dtype) @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            attn_output = cross_attn.out(attn_output)
            x = x + attn_output

            # MLP
            x = x + block.mlp(block.mlp_ln(x))

        x = model.decoder.ln(x)
        logits = (x @ model.decoder.token_embedding.weight.T).float()

        # Get next token
        next_token = logits[0, -1].argmax().item()

        if next_token == tokenizer.eot:
            break

        tokens.append(next_token)

    # Convert to numpy arrays [n_tokens, n_audio_frames]
    attention_matrices = {}
    for key, weights_list in all_attention_weights.items():
        attention_matrices[key] = np.stack(weights_list, axis=0)

    # Decode text
    text = tokenizer.decode(tokens[len(tokenizer.sot_sequence):])

    return attention_matrices, tokens, text


def compute_monotonicity_score(attention: np.ndarray) -> float:
    """
    Compute how monotonically increasing the attention pattern is.

    For each token, find the frame with maximum attention (argmax).
    A good alignment head should have these argmax positions increasing
    monotonically (or nearly so) as tokens progress.

    Returns a score between 0 and 1, where 1 is perfectly monotonic.
    """
    n_tokens, n_frames = attention.shape

    if n_tokens < 2:
        return 0.0

    # Get the frame with maximum attention for each token
    peak_positions = np.argmax(attention, axis=1)

    # Count how many times position increases (or stays same)
    increases = 0
    for i in range(1, len(peak_positions)):
        if peak_positions[i] >= peak_positions[i - 1]:
            increases += 1

    monotonicity = increases / (len(peak_positions) - 1)
    return monotonicity


def compute_diagonal_score(attention: np.ndarray) -> float:
    """
    Compute how diagonal the attention pattern is.

    A diagonal pattern means token i attends mostly to audio frame i*scale,
    where scale = n_frames / n_tokens.
    """
    n_tokens, n_frames = attention.shape

    if n_tokens < 2:
        return 0.0

    # Expected diagonal positions
    scale = n_frames / n_tokens
    expected_positions = np.arange(n_tokens) * scale

    # Actual peak positions
    peak_positions = np.argmax(attention, axis=1)

    # Compute correlation between expected and actual
    if np.std(peak_positions) < 1e-6:
        return 0.0

    correlation = np.corrcoef(expected_positions, peak_positions)[0, 1]

    # Handle NaN
    if np.isnan(correlation):
        return 0.0

    return max(0, correlation)  # Only positive correlations indicate good alignment


def analyze_attention_heads(
    attention_matrices: Dict[Tuple[int, int], np.ndarray],
    top_k: int = 10,
) -> List[Tuple[Tuple[int, int], float, float]]:
    """
    Analyze all attention heads and rank them by alignment quality.

    Returns list of ((layer, head), monotonicity_score, diagonal_score) sorted by combined score.
    """
    results = []

    for (layer, head), attention in attention_matrices.items():
        mono_score = compute_monotonicity_score(attention)
        diag_score = compute_diagonal_score(attention)
        combined_score = (mono_score + diag_score) / 2
        results.append(((layer, head), mono_score, diag_score, combined_score))

    # Sort by combined score (descending)
    results.sort(key=lambda x: x[3], reverse=True)

    return results[:top_k]


def main():
    args = get_args()

    # Check if model needs to be loaded from checkpoint
    model_path = None
    if args.model == "distil-small.en":
        model_path = "distil-small-en-original-model.bin"
        if not os.path.exists(model_path):
            print(f"Downloading {args.model}...")
            import urllib.request
            url = "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/original-model.bin"
            urllib.request.urlretrieve(url, model_path)
    elif args.model == "distil-medium.en":
        model_path = "distil-medium-en-original-model.bin"
        if not os.path.exists(model_path):
            print(f"Downloading {args.model}...")
            import urllib.request
            url = "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/original-model.bin"
            urllib.request.urlretrieve(url, model_path)
    elif args.model == "distil-large-v2":
        model_path = "distil-large-v2-original-model.bin"
        if not os.path.exists(model_path):
            print(f"Downloading {args.model}...")
            import urllib.request
            url = "https://huggingface.co/distil-whisper/distil-large-v2/resolve/main/original-model.bin"
            urllib.request.urlretrieve(url, model_path)
    elif args.model == "distil-large-v3":
        model_path = "distil-large-v3-original-model.bin"
        if not os.path.exists(model_path):
            print(f"Downloading {args.model}...")
            import urllib.request
            url = "https://huggingface.co/distil-whisper/distil-large-v3-openai/resolve/main/model.bin"
            urllib.request.urlretrieve(url, model_path)
    elif args.model == "distil-large-v3.5":
        model_path = "distil-large-v3.5-original-model.bin"
        if not os.path.exists(model_path):
            print(f"Downloading {args.model}...")
            import urllib.request
            url = "https://huggingface.co/distil-whisper/distil-large-v3.5-openai/resolve/main/model.bin"
            urllib.request.urlretrieve(url, model_path)

    # Load model
    print(f"Loading model: {args.model}")
    if model_path:
        model = whisper.load_model(model_path)
    else:
        model = whisper.load_model(args.model)

    print(f"Model dimensions: {model.dims}")

    # Check if model already has alignment heads
    if hasattr(model, 'alignment_heads') and model.alignment_heads is not None:
        try:
            indices = model.alignment_heads.indices()
            existing_heads = list(zip(indices[0].tolist(), indices[1].tolist()))
            print(f"Model has pre-defined alignment heads: {existing_heads}")
        except:
            print("Model has alignment_heads attribute but couldn't parse it")

    # Run transcription and capture attention
    print(f"\nTranscribing: {args.audio}")
    attention_matrices, tokens, text = compute_cross_attention_weights(model, args.audio)

    print(f"\nTranscription: {text}")
    print(f"Number of tokens: {len(tokens)}")

    # Analyze heads
    print(f"\nAnalyzing {len(attention_matrices)} attention heads...")
    top_heads = analyze_attention_heads(attention_matrices, args.top_k)

    print(f"\nTop {args.top_k} alignment head candidates:")
    print("-" * 60)
    print(f"{'Layer':>6} {'Head':>6} {'Monotonic':>12} {'Diagonal':>12} {'Combined':>12}")
    print("-" * 60)

    for (layer, head), mono, diag, combined in top_heads:
        print(f"{layer:>6} {head:>6} {mono:>12.3f} {diag:>12.3f} {combined:>12.3f}")

    # Generate Python code for the best heads
    print("\n" + "=" * 60)
    print("Suggested ALIGNMENT_HEADS entry:")
    print("=" * 60)

    # Use heads with combined score > 0.7 (or top 6 if fewer qualify)
    good_heads = [(l, h) for (l, h), m, d, c in top_heads if c > 0.7]
    if len(good_heads) < 6:
        good_heads = [(l, h) for (l, h), m, d, c in top_heads[:6]]

    model_name = args.model.replace("-", "_").replace(".", "_")
    print(f'"{args.model}": {good_heads},')


if __name__ == "__main__":
    main()
