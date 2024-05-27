#!/usr/bin/env bash

set -ex

if [ ! -d sherpa-onnx-nemo-ctc-en-conformer-medium ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2
  tar xvf sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2
  rm sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2
fi

go mod tidy
go build

./non-streaming-decode-files \
  --nemo-ctc ./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
  --tokens ./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
  --model-type nemo_ctc \
  --debug 0 \
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav
