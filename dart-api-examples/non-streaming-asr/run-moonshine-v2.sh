#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
  tar xvf sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
  rm sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
fi

dart run \
  ./bin/moonshine_v2.dart \
  --encoder ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort \
  --decoder ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/decoder_model_merged.ort \
  --tokens ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/tokens.txt \
  --input-wav ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/test_wavs/0.wav
