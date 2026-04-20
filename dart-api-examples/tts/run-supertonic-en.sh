#!/usr/bin/env bash

set -ex

dart pub get

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/supertonic.html
# to download more models
if [ ! -f ./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
  tar xf sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
  rm sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
fi

dart run \
  ./bin/supertonic-en.dart \
  --duration-predictor ./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx \
  --text-encoder ./sherpa-onnx-supertonic-tts-int8-2026-03-06/text_encoder.int8.onnx \
  --vector-estimator ./sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx \
  --vocoder ./sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx \
  --tts-json ./sherpa-onnx-supertonic-tts-int8-2026-03-06/tts.json \
  --unicode-indexer ./sherpa-onnx-supertonic-tts-int8-2026-03-06/unicode_indexer.bin \
  --voice-style ./sherpa-onnx-supertonic-tts-int8-2026-03-06/voice.bin \
  --sid 6 \
  --speed 1.25 \
  --num-steps 5 \
  --output-wav supertonic-en-0.wav \
  --text "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

ls -lh *.wav
