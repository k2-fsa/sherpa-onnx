#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-zipformer-en-2023-04-01 ]; then
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
  tar xvf sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
  rm sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
fi

dotnet run \
  --tokens=./sherpa-onnx-zipformer-en-2023-04-01/tokens.txt \
  --encoder=./sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx \
  --decoder=./sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx \
  --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  --decoding-method=modified_beam_search \
  --files ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
  ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav \
  ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/8k.wav
