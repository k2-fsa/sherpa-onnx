---
license: apache-2.0
language:
- gl
pipeline_tag: automatic-speech-recognition
---

# Introduction

This directory exports w2v-BERT 2.0 CTC models (transformers'
`Wav2Vec2BertForCTC`) to ONNX for sherpa-onnx.

The default model is
[proxectonos/w2v-bert-2.0-gl](https://huggingface.co/proxectonos/w2v-bert-2.0-gl),
a Galician fine-tune of
[facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) by
Proxecto Nós, licensed under Apache 2.0. Any other `Wav2Vec2BertForCTC`
checkpoint can be exported with `MODEL_ID=<id> ./run.sh` or
`./export_onnx.py --model-id <id>`.

The test waves are the first recordings of the Galician test set of
[FLEURS](https://huggingface.co/datasets/google/fleurs), licensed under
CC BY 4.0.

# Files

- `model.onnx` and `model.weights`: fp32; `model.weights` must stay beside
  `model.onnx`.
- `model.int8.onnx`: MatMul weights quantized to uint8.
- `tokens.txt`

# Inputs and outputs

- `input_features`: float32 `(N, T, 160)`, as transformers'
  `SeamlessM4TFeatureExtractor` computes them: 80 Kaldi filterbank energies
  per 10 ms from samples scaled to 16 bits, each bin normalized to zero mean
  and unit variance over the utterance, and each two consecutive frames
  joined into one row. `compute_features()` in `test_onnx.py` does this
  with kaldi-native-fbank.
- `input_lengths`: int64 `(N,)`, the valid rows of each utterance. Pad a
  batch with zeros; the graph masks the padding.
- `logits`: float32 `(N, T_out, vocab_size)`; the blank is `<blk>`, the
  checkpoint's padding token.
- `logits_lengths`: int64 `(N,)`.

The metadata `subsampling_factor` counts 10 ms filterbank frames per output
frame (4 for the default model: two frames per row, and the adapter's stride
of 2).

# Test

```bash
./run.sh
```

exports the model and decodes the test waves with `test_onnx.py`, which uses
only onnxruntime and kaldi-native-fbank. For `model.onnx` it also decodes
them as one padded batch and checks that the text is the same as decoding
each alone.
