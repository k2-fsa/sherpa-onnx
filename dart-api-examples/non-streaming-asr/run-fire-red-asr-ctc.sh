#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
  tar xvf sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
  rm sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
fi

dart run \
  ./bin/fire-red-asr-ctc.dart \
  --model ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx \
  --tokens ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt \
  --input-wav ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/test_wavs/1.wav
