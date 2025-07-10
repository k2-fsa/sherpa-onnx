#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
  tar xvf sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
  rm sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
fi

for tgt_lang in en de es fr; do
  dart run \
    ./bin/nemo-canary.dart \
    --encoder ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx \
    --decoder ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx \
    --tokens ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt \
    --src-lang en \
    --tgt-lang $tgt_lang \
    --input-wav ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav
done

for tgt_lang in en de; do
  dart run \
    ./bin/nemo-canary.dart \
    --encoder ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx \
    --decoder ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx \
    --tokens ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt \
    --src-lang de \
    --tgt-lang $tgt_lang \
    --input-wav ./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/de.wav
done
