#!/usr/bin/env bash

# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html
# to download the model
# before you run this script.
#
# You can switch to a different offline model if you need

./non-streaming-decode-files \
  --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx \
  --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx \
  --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
  ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav

