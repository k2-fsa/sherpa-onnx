#!/usr/bin/env bash

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/yesno/index.html
# to download the model
# before you run this script.
#

./non-streaming-decode-files \
  --sample-rate=8000 \
  --feat-dim=23 \
  --tokens=./sherpa-onnx-tdnn-yesno/tokens.txt \
  --tdnn-model=./sherpa-onnx-tdnn-yesno/model-epoch-14-avg-2.onnx \
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_0_1_0_0_0_1.wav
