#!/usr/bin/env bash

set -ex

# to download more models
if [ ! -f ./sherpa-onnx-zipvoice-distill-zh-en-emilia/fm_decoder.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
  tar xf sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
  rm sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
fi

go mod tidy
go build

./non-streaming-tts \
  --zipvoice-flow-matching-model sherpa-onnx-zipvoice-distill-zh-en-emilia/fm_decoder.onnx \
  --zipvoice-text-model sherpa-onnx-zipvoice-distill-zh-en-emilia/text_encoder.onnx \
  --zipvoice-data-dir sherpa-onnx-zipvoice-distill-zh-en-emilia/espeak-ng-data \
  --zipvoice-pinyin-dict sherpa-onnx-zipvoice-distill-zh-en-emilia/pinyin.raw \
  --zipvoice-tokens sherpa-onnx-zipvoice-distill-zh-en-emilia/tokens.txt \
  --zipvoice-vocoder sherpa-onnx-zipvoice-distill-zh-en-emilia/vocos_24khz.onnx \
  --prompt-audio sherpa-onnx-zipvoice-distill-zh-en-emilia/prompt.wav \
  --zipvoice-num-steps 4 \
  --num-threads 4 \
  --output-filename=./test-zipvoice.wav \
  --prompt-text "周日被我射熄火了，所以今天是周一。" \
  "我是中国人民的儿子，我爱我的祖国。我得祖国是一个伟大的国家，拥有五千年的文明史。"

