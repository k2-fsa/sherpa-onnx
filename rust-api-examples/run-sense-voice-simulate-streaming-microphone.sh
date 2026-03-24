#!/usr/bin/env bash
set -ex

# https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html
if [ ! -f "./silero_vad.onnx" ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

# see
# https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html#sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-chinese-english-japanese-korean-cantonese
if [ ! -f ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx ]; then
  curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
  ls -lh sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
fi

cargo run --example sense_voice_simulate_streaming_microphone --features mic -- \
    --silero-vad-model ./silero_vad.onnx \
    --model ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx \
    --tokens ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt
