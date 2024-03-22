#!/usr/bin/env bash
set -ex
if [ ! -f ./vits-zh-aishell3/vits-aishell3.onnx ]; then
  # wget -qq https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2
  curl -OL https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2
  tar xf vits-zh-aishell3.tar.bz2
  rm vits-zh-aishell3.tar.bz2
fi

dotnet run \
  --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
  --vits-tokens=./vits-zh-aishell3/tokens.txt \
  --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
  --tts-rule-fsts=./vits-zh-aishell3/rule.fst \
  --sid=66 \
  --debug=1 \
  --output-filename=./aishell3-66.wav \
  --text="这是一个语音合成测试, 写于公元 2024 年 1 月 28 号, 23点27分，星期天。"
