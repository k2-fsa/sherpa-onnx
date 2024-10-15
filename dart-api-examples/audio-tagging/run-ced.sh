#!/usr/bin/env bash

set -ex

dart pub get

if [[ ! -f ./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/model.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2
  tar xvf sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2
  rm sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2
fi

for w in 1 2 3 4 5 6; do
  dart run \
    ./bin/ced.dart \
    --model ./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/model.int8.onnx \
    --labels ./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/class_labels_indices.csv \
    --wav ./sherpa-onnx-ced-mini-audio-tagging-2024-04-19/test_wavs/$w.wav
done
