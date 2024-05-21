#!/usr/bin/env bash
set -ex

if [ ! -d vits-icefall-zh-aishell3 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
  tar xvf vits-icefall-zh-aishell3.tar.bz2
  rm vits-icefall-zh-aishell3.tar.bz2
fi

go mod tidy
go build

for sid in 10 33 99; do
./non-streaming-tts \
  --vits-model=./vits-icefall-zh-aishell3/model.onnx \
  --vits-lexicon=./vits-icefall-zh-aishell3/lexicon.txt \
  --vits-tokens=./vits-icefall-zh-aishell3/tokens.txt \
  --sid=$sid \
  --debug=1 \
  --output-filename=./liliana-$sid.wav \
  "林美丽最美丽、最漂亮、最可爱！"

./non-streaming-tts \
  --vits-model=./vits-icefall-zh-aishell3/model.onnx \
  --vits-lexicon=./vits-icefall-zh-aishell3/lexicon.txt \
  --vits-tokens=./vits-icefall-zh-aishell3/tokens.txt \
  --tts-rule-fsts=./vits-icefall-zh-aishell3/phone.fst,./vits-icefall-zh-aishell3/date.fst,./vits-icefall-zh-aishell3/number.fst \
  --sid=$sid \
  --debug=1 \
  --output-filename=./numbers-$sid.wav \
  "数字12345.6789怎么念"

./non-streaming-tts \
  --vits-model=./vits-icefall-zh-aishell3/model.onnx \
  --vits-lexicon=./vits-icefall-zh-aishell3/lexicon.txt \
  --vits-tokens=./vits-icefall-zh-aishell3/tokens.txt \
  --tts-rule-fsts=./vits-icefall-zh-aishell3/phone.fst,./vits-icefall-zh-aishell3/date.fst,./vits-icefall-zh-aishell3/number.fst \
  --tts-rule-fars=./vits-icefall-zh-aishell3/rule.far \
  --sid=$sid \
  --debug=1 \
  --output-filename=./heteronym-$sid.wav \
  "万古长存长沙长大长白山长孙长安街"
done
