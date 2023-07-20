#!/usr/bin/env bash

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/nemo/english.html#stt-en-conformer-ctc-medium
# to download the model
# before you run this script.
#
# You can switch to a different online model if you need

./non-streaming-decode-files \
  --nemo-ctc ./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
  --tokens ./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
  --model-type nemo_ctc \
  --debug 0 \
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav
