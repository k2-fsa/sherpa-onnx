#!/usr/bin/env bash

set -ex

dart pub get

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html
# to download more models
if [ ! -f ./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  tar xvf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
fi

dart run \
  ./bin/pocket-en.dart \
  --lm-flow ./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx \
  --lm-main ./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx \
  --encoder ./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx \
  --decoder ./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx \
  --text-conditioner ./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx \
  --vocab-json ./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json \
  --token-scores-json ./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json \
  --reference-audio ./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav \
  --output-wav pocket-en-0.wav \
  --text "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

ls -lh *.wav
