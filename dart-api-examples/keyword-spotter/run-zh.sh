#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
  tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
  rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
fi

dart run \
  ./bin/zipformer-transducer.dart \
  --encoder ./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
  --decoder ./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx \
  --joiner ./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx \
  --tokens ./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt \
  --keywords-file ./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt \
  --input-wav ./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav

