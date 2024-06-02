#!/usr/bin/env bash

set -ex

if [ ! -d icefall-asr-zipformer-streaming-wenetspeech-20230615 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/icefall-asr-zipformer-streaming-wenetspeech-20230615.tar.bz2
  tar xvf icefall-asr-zipformer-streaming-wenetspeech-20230615.tar.bz2
  rm icefall-asr-zipformer-streaming-wenetspeech-20230615.tar.bz2
fi

go mod tidy
go build

./real-time-speech-recognition-from-microphone \
  --encoder ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx \
  --decoder ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx \
  --joiner ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx \
  --tokens ./icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
  --model-type zipformer2
