#!/usr/bin/env bash

set -ex

if [ ! -f mel_spec_22khz_univ.onnx ]; then
  curl -SL -O https://huggingface.co/BSC-LT/vocos-mel-22khz/resolve/main/mel_spec_22khz_univ.onnx
fi

if [ ! -f ./vocos-22khz-univ.onnx ]; then
  python3 ./add_meta_data.py --in-model ./mel_spec_22khz_univ.onnx --out-model ./vocos-22khz-univ.onnx
fi

# The following is for testing
if [ ! -f ./matcha-icefall-en_US-ljspeech/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
  tar xf matcha-icefall-en_US-ljspeech.tar.bz2
  rm matcha-icefall-en_US-ljspeech.tar.bz2
fi

if [ ! -f ./hifigan_v2.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx
fi

python3 ./test.py
ls -lh
