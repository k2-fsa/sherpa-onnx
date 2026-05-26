#!/usr/bin/env bash
set -ex

# https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html
if [ ! -f "./silero_vad.onnx" ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

# see
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/zipformer-ctc-models.html#sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03-chinese
if [ ! -f ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx ]; then
  curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

  tar xvf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
  rm sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
  ls -lh sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03
fi

cargo run --example zipformer_ctc_simulate_streaming_microphone --features mic -- \
    --silero-vad-model ./silero_vad.onnx \
    --model ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx \
    --tokens ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt
