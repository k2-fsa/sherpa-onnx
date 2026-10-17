#!/usr/bin/env bash

set -ex

if [ ! -f ./sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
  tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
  rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
fi

dotnet run -c Release
