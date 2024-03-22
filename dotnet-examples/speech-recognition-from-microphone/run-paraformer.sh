#!/usr/bin/env bash

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-streaming-paraformer-bilingual-zh-en-chinese-english
# to download the model files

set -ex
if [ ! -d ./sherpa-onnx-streaming-paraformer-bilingual-zh-en ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en
  cd sherpa-onnx-streaming-paraformer-bilingual-zh-en
  git lfs pull --include "*.onnx"
  cd ..
fi

dotnet run -c Release \
  --tokens ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt \
  --paraformer-encoder ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx \
  --paraformer-decoder ./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx \
