# Introduction

This folder contains scripts for adding meta data to the
[VoxLock](https://github.com/arbazsiddiqui/voxlock) ONNX models from
<https://huggingface.co/arbazsiddiqui/voxlock/tree/main/onnx>.

VoxLock is a target-speaker voice isolation model. It keeps the voice of one
speaker, e.g., the caller in a phone call, and removes the other voices.
It has two models:

- `voxlock_enroll.onnx` runs once on a voice sample of the caller
  (about 3 seconds at 16 kHz)
- `voxlock_chunk*.onnx` runs on each chunk of the audio and carries the
  LSTM state from chunk to chunk

| Precision | Enroll model | Chunk model |
|---|---|---|
| int8 (LSTMs in 8-bit) | `voxlock_enroll.onnx` | `voxlock_chunk_int8.onnx` |
| int8-full (LSTMs and matrix multiplies in 8-bit) | `voxlock_enroll.onnx` | `voxlock_chunk_int8_full.onnx` |
| fp16 (float16 weights) | `voxlock_enroll_fp16.onnx` | `voxlock_chunk_fp16.onnx` |
| fp32 | `voxlock_enroll.onnx` | `voxlock_chunk.onnx` |

The models with meta data can be downloaded from
<https://github.com/arbazsiddiqui/voxlock/releases/tag/sherpa-onnx-models>

- [./run.sh](./run.sh) downloads the models and runs
  [./add_meta_data.py](./add_meta_data.py), which also checks that the outputs
  of each model do not change
- [./test.py](./test.py) compares the output of `sherpa-onnx-offline-denoiser`
  with the [voxlock](https://pypi.org/project/voxlock/) Python package
