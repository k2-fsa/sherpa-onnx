#!/usr/bin/env bash
set -ex
if [ ! -f ./vits-zh-hf-fanchen-C/vits-zh-hf-fanchen-C.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-hf-fanchen-C.tar.bz2
  tar xf vits-zh-hf-fanchen-C.tar.bz2
  rm vits-zh-hf-fanchen-C.tar.bz2
fi

dotnet run \
  --vits-model=./vits-zh-hf-fanchen-C/vits-zh-hf-fanchen-C.onnx \
  --tokens=./vits-zh-hf-fanchen-C/tokens.txt \
  --lexicon=./vits-zh-hf-fanchen-C/lexicon.txt \
  --tts-rule-fsts=./vits-zh-hf-fanchen-C/phone.fst,./vits-zh-hf-fanchen-C/date.fst,./vits-zh-hf-fanchen-C/number.fst \
  --vits-dict-dir=./vits-zh-hf-fanchen-C/dict \
  --sid=100 \
  --debug=1 \
  --output-filename=./fanchen-100.wav \
  --text="这是一个语音合成测试, 写于公元2024年4月26号, 11点05分，星期5。小米的使命是，始终坚持做'感动人心、价格厚道'的好产品，让全球每个人都能享受科技带来的美好生活。"
