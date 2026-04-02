#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  tar xvf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  rm sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
fi

dart run \
  ./bin/qwen3-asr.dart \
  --conv-frontend ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
  --encoder ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
  --decoder ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
  --tokenizer ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
  --input-wav ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav
