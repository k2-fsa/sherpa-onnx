# Introduction

## Download VAD models

Please download
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
and put `silero_vad.onnx` into the current directory, i.e., `wasm/vad/assets`.

## Download non-streaming ASR models

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
to download a non-streaming ASR model, i.e., an offline ASR model.

After downloading, you should rename the model files.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/lazarus/generate-subtitles.html#download-a-speech-recognition-model
for how to rename.

You can find example build scripts at the following address:

  https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/wasm-simd-hf-space-vad-asr.yaml
