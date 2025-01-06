#!/usr/bin/env bash

set -ex

dart pub get

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-zh-baker-chinese-1-female-speaker
# to download more models
if [ ! -f ./matcha-icefall-zh-baker/model-steps-3.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
  tar xvf matcha-icefall-zh-baker.tar.bz2
  rm matcha-icefall-zh-baker.tar.bz2
fi

if [ ! -f ./hifigan_v2.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx
fi

dart run \
  ./bin/matcha-zh.dart \
  --acoustic-model ./matcha-icefall-zh-baker/model-steps-3.onnx \
  --vocoder ./hifigan_v2.onnx \
  --lexicon ./matcha-icefall-zh-baker/lexicon.txt \
  --tokens ./matcha-icefall-zh-baker/tokens.txt \
  --dict-dir ./matcha-icefall-zh-baker/dict \
  --rule-fsts ./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst \
  --sid 0 \
  --speed 1.0 \
  --output-wav matcha-zh-1.wav \
  --text "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。" \

dart run \
  ./bin/matcha-zh.dart \
  --acoustic-model ./matcha-icefall-zh-baker/model-steps-3.onnx \
  --vocoder ./hifigan_v2.onnx \
  --lexicon ./matcha-icefall-zh-baker/lexicon.txt \
  --tokens ./matcha-icefall-zh-baker/tokens.txt \
  --dict-dir ./matcha-icefall-zh-baker/dict \
  --sid 0 \
  --speed 1.0 \
  --output-wav matcha-zh-2.wav \
  --text "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔." \

ls -lh *.wav
