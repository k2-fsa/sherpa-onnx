# Export Qwen3-ForcedAligner to ONNX

This directory contains the scripts used to export
[Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
to ONNX, for use with the optional word-level timestamp support of the
Qwen3-ASR recognizer.

It produces three files, following the same layout as the qwen3-asr models:

| File                  | Inputs                                        | Output                 |
|-----------------------|-----------------------------------------------|------------------------|
| `conv_frontend.onnx`  | `input_features` (B,T,128) mel                | `conv_output` (B,A,1024)|
| `encoder.onnx`        | `input_features`, `feature_attention_mask`    | `audio_features` (B,A,1024) |
| `decoder.onnx`        | `input_ids`, `audio_features`, `attention_mask` | `logits` (B,S,5000)  |

The decoder is a single forward pass with no KV cache: it reads the ASR
transcript interleaved with `<|timestamp|>` slots and classifies each slot
into one of 5000 timestamp classes (80 ms per class).

## Requirements

```bash
pip install -U qwen-asr onnx onnxruntime soundfile
```

Alternatively, clone https://github.com/QwenLM/Qwen3-ASR and pass
`--qwen-asr-repo /path/to/Qwen3-ASR`.

## Export

```bash
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B \
    --local-dir ./qwen3-forced-aligner

./export-onnx.py \
    --model ./qwen3-forced-aligner \
    --outdir ./out
```

The aligner has its own tokenizer (`vocab.json`, `merges.txt`,
`tokenizer_config.json` in the model directory). It differs from the ASR
tokenizer because it contains the `<|timestamp|>` token, so pass the model
directory itself via `--qwen3-asr-forced-aligner-tokenizer`.

## Verify

`test-onnx.py` runs the exported ONNX pipeline and the original PyTorch
model on the same audio/text pair and compares per-word timestamp slots:

```bash
./test-onnx.py \
    --model ./qwen3-forced-aligner \
    --onnx-dir ./out \
    --wav ./test.wav \
    --lang English \
    --text "the transcript produced by ASR"
```

Important: the text must be the transcript produced by ASR. Feeding an
independent reference transcript is meaningless — the aligner aligns the
words it is given, so words that were not recognized cannot be aligned.
Japanese verification additionally needs `pip install nagisa`; Korean needs
`pip install soynlp`. Other languages work without them.

## Notes on the encoder

The reference implementation's audio encoder computes chunk metadata
(`cu_seqlens`) that only takes effect with `flash_attention_2`. Under the
default eager/SDPA attention implementations the argument is ignored and
`_prepare_attention_mask` is never called, so the reference behavior is
full bidirectional attention over the valid audio tokens. `encoder.py`
reproduces that behavior so the exported model matches the reference
bit-for-bit (timestamp logits max diff ~2e-5 in our tests).
