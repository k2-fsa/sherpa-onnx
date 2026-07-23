#!/usr/bin/env bash
set -ex

# see
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
if [ ! -f ./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
fi

cargo run --example funasr_nano -- \
    --wav ./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_yue.wav \
    --encoder-adaptor ./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx \
    --embedding ./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx \
    --llm ./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx \
    --tokenizer ./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B \
    --num-threads 2
