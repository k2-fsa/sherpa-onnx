#!/usr/bin/env bash

set -ex

dart pub get


# Please visit
# https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# to download more models

if [[ ! -f ./sherpa-onnx-vits-zh-ll/tokens.txt ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2
  tar xvf sherpa-onnx-vits-zh-ll.tar.bz2
  rm sherpa-onnx-vits-zh-ll.tar.bz2
fi

dart run \
  ./bin/zh.dart \
  --model ./sherpa-onnx-vits-zh-ll/model.onnx \
  --lexicon ./sherpa-onnx-vits-zh-ll/lexicon.txt \
  --tokens ./sherpa-onnx-vits-zh-ll/tokens.txt \
  --dict-dir ./sherpa-onnx-vits-zh-ll/dict \
  --sid 2 \
  --speed 1.0 \
  --text '当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔。' \
  --output-wav zh-jieba-2.wav

dart run \
  ./bin/zh.dart \
  --model ./sherpa-onnx-vits-zh-ll/model.onnx \
  --lexicon ./sherpa-onnx-vits-zh-ll/lexicon.txt \
  --tokens ./sherpa-onnx-vits-zh-ll/tokens.txt \
  --dict-dir ./sherpa-onnx-vits-zh-ll/dict \
  --rule-fsts "./sherpa-onnx-vits-zh-ll/phone.fst,./sherpa-onnx-vits-zh-ll/date.fst,./sherpa-onnx-vits-zh-ll/number.fst" \
  --sid 3 \
  --speed 1.0 \
  --text '今天是2024年6月15号，13点23分。如果有困难，请拨打110或者18920240511。123456块钱。' \
  --output-wav zh-jieba-3.wav

ls -lh *.wav
