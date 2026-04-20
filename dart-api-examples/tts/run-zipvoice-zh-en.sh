#!/usr/bin/env bash

set -ex

dart pub get

# please visit
# https://k2-fsa.github.io/sherpa/onnx/tts/zipvoice.html
# to download more models
if [ ! -f ./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
  tar xvf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
  rm sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
fi

if [ ! -f ./vocos_24khz.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx
fi

dart run \
  ./bin/zipvoice-zh-en.dart \
  --tokens ./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt \
  --encoder ./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx \
  --decoder ./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx \
  --vocoder ./vocos_24khz.onnx \
  --data-dir ./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data \
  --lexicon ./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt \
  --reference-audio ./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav \
  --reference-text "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系." \
  --num-steps 4 \
  --output-wav zipvoice-zh-en-0.wav \
  --text "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中."

ls -lh *.wav
