#!/usr/bin/env bash

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-streaming-paraformer-bilingual-zh-en-chinese-english
# to download the model files

set -ex
if [ ! -d ./sherpa-onnx-streaming-paraformer-bilingual-zh-en ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
  tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
  rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
fi

dotnet run -c Release \
  --tokens ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt \
  --paraformer-encoder ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx \
  --paraformer-decoder ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx \
