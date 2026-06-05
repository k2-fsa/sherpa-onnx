#!/usr/bin/env bash
set -ex

# see
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
if [ ! -f ./sherpa-onnx-whisper-tiny/tiny-encoder.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.tar.bz2
  rm sherpa-onnx-whisper-tiny.tar.bz2
fi

cargo run --example whisper -- \
    --wav ./sherpa-onnx-whisper-tiny/test_wavs/0.wav \
    --encoder ./sherpa-onnx-whisper-tiny/tiny-encoder.onnx \
    --decoder ./sherpa-onnx-whisper-tiny/tiny-decoder.onnx \
    --tokens ./sherpa-onnx-whisper-tiny/tiny-tokens.txt \
    --language en \
    --num-threads 2
