#!/usr/bin/env bash
set -ex

# see
# https://k2-fsa.github.io/sherpa/onnx/moonshine
if [ ! -f ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
  tar xvf sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
  rm sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
fi

cargo run --example moonshine_v2 -- \
    --wav ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/test_wavs/0.wav \
    --encoder ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort \
    --decoder ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/decoder_model_merged.ort \
    --tokens ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/tokens.txt \
    --num-threads 2
