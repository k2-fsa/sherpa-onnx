#!/usr/bin/env bash
set -ex

# https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html
if [ ! -f "./silero_vad.onnx" ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
fi

if [ ! -f ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx ]; then
  curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2

  tar xvf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  rm sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  ls -lh sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
fi

cargo run --example qwen3_asr_simulate_streaming_microphone --features mic -- \
    --silero-vad-model ./silero_vad.onnx \
    --conv-frontend ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
    --encoder ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
    --decoder ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
    --tokenizer ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer
