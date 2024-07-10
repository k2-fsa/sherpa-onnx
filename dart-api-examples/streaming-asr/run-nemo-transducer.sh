#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
  tar xvf sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
  rm sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
fi

dart run \
  ./bin/zipformer-transducer.dart \
  --encoder ./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/encoder.onnx \
  --decoder ./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/decoder.onnx \
  --joiner ./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/joiner.onnx \
  --tokens ./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/tokens.txt \
  --input-wav ./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/test_wavs/0.wav
