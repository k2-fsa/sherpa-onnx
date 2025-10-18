#!/usr/bin/env bash

set -ex

# Download ZipVoice model pack if missing
if [ ! -f ./sherpa-onnx-zipvoice-distill-zh-en-emilia/fm_decoder.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
  tar xf sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
  rm sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
fi

dotnet build

dotnet run \
  --zipvoice-flow-matching-model=./sherpa-onnx-zipvoice-distill-zh-en-emilia/fm_decoder.onnx \
  --zipvoice-text-model=./sherpa-onnx-zipvoice-distill-zh-en-emilia/text_encoder.onnx \
  --tokens=./sherpa-onnx-zipvoice-distill-zh-en-emilia/tokens.txt \
  --zipvoice-data-dir=./sherpa-onnx-zipvoice-distill-zh-en-emilia/espeak-ng-data \
  --zipvoice-pinyin-dict=./sherpa-onnx-zipvoice-distill-zh-en-emilia/pinyin.raw \
  --zipvoice-vocoder=./sherpa-onnx-zipvoice-distill-zh-en-emilia/vocos_24khz.onnx \
  --prompt-audio=./sherpa-onnx-zipvoice-distill-zh-en-emilia/prompt.wav \
  --prompt-text="周日被我射熄火了，所以今天是周一。" \
  --zipvoice-num-steps=4 \
  --debug=1 \
  --output-filename=./test-zipvoice.wav \
  --text="我是中国人民的儿子，我爱我的祖国。我得祖国是一个伟大的国家，拥有五千年的文明史。"