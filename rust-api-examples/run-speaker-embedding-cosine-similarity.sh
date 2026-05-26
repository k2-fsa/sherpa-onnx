#!/usr/bin/env bash
set -ex

if [ ! -f ./wespeaker_zh_cnceleb_resnet34.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_zh_cnceleb_resnet34.onnx
fi

if [ ! -f ./fangjun-sr-1.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/fangjun-sr-1.wav
fi

if [ ! -f ./fangjun-sr-2.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/fangjun-sr-2.wav
fi

if [ ! -f ./leijun-sr-1.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/leijun-sr-1.wav
fi

cargo run --example speaker_embedding_cosine_similarity
