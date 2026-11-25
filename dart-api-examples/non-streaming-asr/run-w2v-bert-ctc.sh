#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
  tar xvf sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
  rm sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
fi

dart run \
  ./bin/w2v-bert-ctc.dart \
  --model ./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/model.int8.onnx \
  --tokens ./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/tokens.txt \
  --input-wav ./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/test_wavs/0.wav
