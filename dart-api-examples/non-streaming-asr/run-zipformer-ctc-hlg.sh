#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2

  tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
  rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
fi

dart run \
  ./bin/zipformer-ctc-hlg.dart \
  --model ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx \
  --tokens ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt \
  --graph ./sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst \
  --input-wav ./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/0.wav
