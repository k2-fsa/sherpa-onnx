#!/usr/bin/env bash

# please refer to
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#aishell3-chinese-multi-speaker-174-speakers
# to download the model before you run this script

for sid in 10 33 99; do
./non-streaming-tts \
  --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
  --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
  --vits-tokens=./vits-zh-aishell3/tokens.txt \
  --sid=$sid \
  --debug=1 \
  --output-filename=./liliana-$sid.wav \
  "林美丽最美丽、最漂亮、最可爱！"

./non-streaming-tts \
  --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
  --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
  --vits-tokens=./vits-zh-aishell3/tokens.txt \
  --tts-rule-fsts=./vits-zh-aishell3/rule.fst \
  --sid=$sid \
  --debug=1 \
  --output-filename=./numbers-$sid.wav \
  "数字12345.6789怎么念"
done
