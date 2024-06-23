#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
  tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
  rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
fi

if [ ! -f ./itn-zh-number.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
fi

if [ ! -f ./itn_zh_number.fst ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
fi

dart run \
  ./bin/zipformer-transducer.dart \
  --encoder ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx \
  --decoder ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
  --joiner ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx \
  --tokens ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
  --rule-fsts ./itn_zh_number.fst \
  --input-wav ./itn-zh-number.wav
