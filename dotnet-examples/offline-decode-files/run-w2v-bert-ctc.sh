#!/usr/bin/env bash

set -ex

if [ ! -f ./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/model.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
  tar xvf sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
  rm sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2

  ls -lh sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20
fi

dotnet run \
  --num-threads=2 \
  --w2v-bert-ctc=./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/model.int8.onnx \
  --tokens=./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/tokens.txt \
  --files ./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/test_wavs/0.wav
