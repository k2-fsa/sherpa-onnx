#!/usr/bin/env bash
set -ex
if [ ! -f ./vits-zh-aishell3/vits-aishell3.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
  tar xvf vits-icefall-zh-aishell3.tar.bz2
  rm vits-icefall-zh-aishell3.tar.bz2
fi

dotnet run \
  --vits-model=./vits-icefall-zh-aishell3/model.onnx \
  --tokens=./vits-icefall-zh-aishell3/tokens.txt \
  --lexicon=./vits-icefall-zh-aishell3/lexicon.txt \
  --tts-rule-fsts=./vits-icefall-zh-aishell3/phone.fst,./vits-icefall-zh-aishell3/date.fst,./vits-icefall-zh-aishell3/number.fst \
  --tts-rule-fars=./vits-icefall-zh-aishell3/rule.far \
  --sid=66 \
  --debug=1 \
  --output-filename=./aishell3-66.wav \
  --text="这是一个语音合成测试, 写于公元 2024 年 1 月 28 号, 23点27分，星期天。长沙长大，去过长白山和长安街。行行出状元。行行，银行行长，行业。"
