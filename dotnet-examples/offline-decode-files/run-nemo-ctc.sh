#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-nemo-ctc-en-conformer-medium ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2
  tar xvf sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2
  rm sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2
fi

dotnet run \
  --tokens=./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
  --nemo-ctc=./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
  --num-threads=1 \
  --files ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav
