# Introduction

This folder contains code showing how to convert [Whisper][whisper] to onnx
and use onnxruntime to replace PyTorch for speech recognition.

You can use [sherpa-onnx][sherpa-onnx] to run the converted model.

Please see
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/export-onnx.html
for details.

## Finding Alignment Heads for Word Timestamps

The `export-onnx-with-attention.py` script exports Whisper models with
cross-attention weights for word-level timestamps. It requires knowing which
attention heads are "alignment heads" - heads that show monotonically increasing
attention patterns useful for aligning audio to text.

For standard OpenAI Whisper models, alignment heads are defined in the
`ALIGNMENT_HEADS` dict in the export script. For new or custom models (like
distil-whisper variants), you can discover alignment heads using:

```bash
python find_alignment_heads.py --model <model-name> --audio <test-audio.wav>
```

This script analyzes all attention heads and ranks them by:
- **Monotonicity**: Whether attention peaks move forward as tokens are decoded
- **Diagonal score**: Correlation with expected diagonal attention pattern

Example output:
```
Top 15 alignment head candidates:
------------------------------------------------------------
 Layer   Head    Monotonic     Diagonal     Combined
------------------------------------------------------------
     3      2        0.846        0.985        0.915
     0      0        0.962        0.617        0.789
     ...
```

Heads with high combined scores (>0.7) are good candidates. A single head with
a very high diagonal score (>0.9) is often sufficient for accurate timestamps.

[whisper]: https://github.com/openai/whisper
[sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
