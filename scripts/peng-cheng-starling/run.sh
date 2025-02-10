#!/usr/bin/env bash

set -ex

if [ ! -f bpe.model ]; then
  curl -SL -O https://huggingface.co/stdo/PengChengStarling/resolve/main/bpe.model
fi

if [ ! -f tokens.txt ]; then
  curl -SL -O https://huggingface.co/stdo/PengChengStarling/resolve/main/tokens.txt
fi

if [ ! -f decoder-epoch-75-avg-11-chunk-16-left-128.onnx ]; then
  curl -SL -O https://huggingface.co/stdo/PengChengStarling/resolve/main/decoder-epoch-75-avg-11-chunk-16-left-128.onnx
fi

if [ ! -f encoder-epoch-75-avg-11-chunk-16-left-128.onnx ]; then
  curl -SL -O https://huggingface.co/stdo/PengChengStarling/resolve/main/encoder-epoch-75-avg-11-chunk-16-left-128.onnx
fi

if [ ! -f joiner-epoch-75-avg-11-chunk-16-left-128.onnx ]; then
  curl -SL -O https://huggingface.co/stdo/PengChengStarling/resolve/main/joiner-epoch-75-avg-11-chunk-16-left-128.onnx
fi

mkdir -p test_wavs
if [ ! -f test_wavs/zh.wav ]; then
  curl -SL --output test_wavs/zh.wav https://huggingface.co/marcoyang/sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23/resolve/main/test_wavs/0.wav
fi

if [ ! -f test_wavs/en.wav ]; then
  curl -SL --output test_wavs/en.wav https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21/resolve/main/test_wavs/0.wav
fi

if [ ! -f test_wavs/ja.wav ]; then
  curl -SL --output test_wavs/ja.wav https://huggingface.co/csukuangfj/reazonspeech-k2-v2/resolve/main/test_wavs/5.wav
fi
