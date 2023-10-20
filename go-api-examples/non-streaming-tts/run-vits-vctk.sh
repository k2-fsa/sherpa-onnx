#!/usr/bin/env bash

# please refer to
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#vctk-english-multi-speaker-109-speakers
# to download the model before you run this script

for sid in 0 10 108; do
./non-streaming-tts \
  --vits-model=./vits-vctk/vits-vctk.onnx \
  --vits-lexicon=./vits-vctk/lexicon.txt \
  --vits-tokens=./vits-vctk/tokens.txt \
  --sid=0 \
  --debug=1 \
  --output-filename=./kennedy-$sid.wav \
  'Ask not what your country can do for you; ask what you can do for your country.'
done
