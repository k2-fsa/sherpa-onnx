# Introduction

This folder contains C API examples for [sherpa-onnx][sherpa-onnx].

Please refer to the documentation
https://k2-fsa.github.io/sherpa/onnx/c-api/index.html
for details.


## File descriptions

- [decode-file-c-api.c](./decode-file-c-api.c) This file shows how to use the C API
  for speech recognition with a streaming model.

- [offline-tts-c-api.c](./offline-tts-c-api.c) This file shows how to use the C API
  to convert text to speech with a non-streaming model.

- [speech-enhancement-gtcrn-c-api.c](./speech-enhancement-gtcrn-c-api.c)
  This file shows how to use the C API for speech enhancement with GTCRN
  models.

- [speech-enhancement-dpdfnet-c-api.c](./speech-enhancement-dpdfnet-c-api.c)
  This file shows how to use the C API for speech enhancement with DPDFNet
  models. Use 16 kHz DPDFNet models such as `dpdfnet_baseline.onnx`,
  `dpdfnet2.onnx`, `dpdfnet4.onnx`, or `dpdfnet8.onnx` for downstream ASR and
  `dpdfnet2_48khz_hr.onnx` for 48 kHz enhancement output.

- [online-speech-enhancement-gtcrn-c-api.c](./online-speech-enhancement-gtcrn-c-api.c)
  This file shows how to use the C API for online speech enhancement with
  GTCRN models.

- [online-speech-enhancement-dpdfnet-c-api.c](./online-speech-enhancement-dpdfnet-c-api.c)
  This file shows how to use the C API for online speech enhancement with
  DPDFNet models. Use `dpdfnet_baseline.onnx`, `dpdfnet2.onnx`,
  `dpdfnet4.onnx`, or `dpdfnet8.onnx` for 16 kHz output.

[sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
