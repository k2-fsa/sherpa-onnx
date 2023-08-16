#!/usr/bin/env bash

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-paraformer-zh-2023-03-28-chinese
# to download the model
# before you run this script.
#

./non-streaming-decode-files \
  --paraformer ./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx \
  --tokens ./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
  --model-type paraformer \
  --debug 0 \
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav
