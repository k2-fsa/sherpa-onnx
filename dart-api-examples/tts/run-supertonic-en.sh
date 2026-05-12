#!/usr/bin/env bash

set -ex

dart pub get

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/supertonic.html
# to download more models
if [ ! -f ./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
  tar xf sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
  rm sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
fi

dart run \
  ./bin/supertonic-en.dart \
  --duration-predictor ./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx \
  --text-encoder ./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx \
  --vector-estimator ./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx \
  --vocoder ./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx \
  --tts-json ./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json \
  --unicode-indexer ./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin \
  --voice-style ./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin \
  --sid 6 \
  --speed 1.25 \
  --num-steps 8 \
  --output-wav supertonic-en-0.wav \
  --text "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

ls -lh *.wav
