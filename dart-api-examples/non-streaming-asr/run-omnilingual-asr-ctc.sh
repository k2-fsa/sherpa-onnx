#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tokens.txt ]; then
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
  tar xvf sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
  rm sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
fi

dart run \
  ./bin/omnilingual-asr-ctc.dart \
  --model ./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/model.int8.onnx \
  --tokens ./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tokens.txt \
  --input-wav ./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/test_wavs/en.wav
