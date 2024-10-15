#!/usr/bin/env bash

set -ex

dart pub get

if [[ ! -f ./sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.onnx ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
  tar xvf sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
  rm sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
fi

for w in 1 2 3 4 5 6; do
  dart run \
    ./bin/zipformer.dart \
    --model ./sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.int8.onnx \
    --labels ./sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv \
    --wav ./sherpa-onnx-zipformer-audio-tagging-2024-04-09/test_wavs/$w.wav
done
