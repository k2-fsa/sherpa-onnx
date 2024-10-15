#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2

  tar xvf sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
  rm sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
fi

dart run \
  ./bin/nemo-transducer.dart \
  --encoder ./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/encoder.onnx \
  --decoder ./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/decoder.onnx \
  --joiner ./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/joiner.onnx \
  --tokens ./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/tokens.txt \
  --input-wav ./sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k/test_wavs/de-german.wav
