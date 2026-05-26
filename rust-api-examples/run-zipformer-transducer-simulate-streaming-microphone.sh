#!/usr/bin/env bash
set -ex

# https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html
if [ ! -f "./silero_vad.onnx" ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

# see
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#sherpa-onnx-zipformer-multi-zh-hans-2023-9-2-chinese
if [ ! -f ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/encoder-epoch-20-avg-1.int8.onnx ]; then
  curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2

  tar xvf sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2
  rm sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2
  ls -lh sherpa-onnx-zipformer-multi-zh-hans-2023-9-2
fi

cargo run --example zipformer_transducer_simulate_streaming_microphone --features mic -- \
    --silero-vad-model ./silero_vad.onnx \
    --encoder ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/encoder-epoch-20-avg-1.int8.onnx \
    --decoder ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/decoder-epoch-20-avg-1.onnx \
    --joiner ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/joiner-epoch-20-avg-1.int8.onnx \
    --tokens ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/tokens.txt
