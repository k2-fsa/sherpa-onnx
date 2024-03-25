#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-zipformer-en-2023-04-01 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
  tar xvf sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
  rm sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
fi

if [ ! -f ./sherpa-onnx-zipformer-en-2023-04-01/hotwords_en.txt ]; then
cat >./sherpa-onnx-zipformer-en-2023-04-01/hotwords_en.txt <<EOF
▁ QUA R TER S
▁FOR E VER
EOF
fi

dotnet run \
  --tokens=./sherpa-onnx-zipformer-en-2023-04-01/tokens.txt \
  --encoder=./sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx \
  --decoder=./sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx \
  --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  --decoding-method=modified_beam_search \
  --files ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
  ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav

dotnet run \
  --hotwords-file=./sherpa-onnx-zipformer-en-2023-04-01/hotwords_en.txt \
  --hotwords-score=2.0 \
  --tokens=./sherpa-onnx-zipformer-en-2023-04-01/tokens.txt \
  --encoder=./sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx \
  --decoder=./sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx \
  --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  --decoding-method=modified_beam_search \
  --files ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
  ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav

# 0.wav: QUARTER -> QUARTERS
# 1.wav: FOR EVER -> FOREVER
