#!/usr/bin/env bash

set -ex

if [ ! -f ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

  tar xvf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
  rm sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
fi

dotnet run \
  --tokens=./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt \
  --zipformer-ctc=./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx \
  --num-threads=1 \
  --files ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/test_wavs/0.wav \
  ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/test_wavs/1.wav \
  ./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/test_wavs/8k.wav
