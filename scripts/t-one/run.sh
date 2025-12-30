#!/usr/bin/env bash

set -ex

if [ ! -f ./model.onnx ]; then
  curl -SL -O https://hf-mirror.com/t-tech/T-one/resolve/main/model.onnx
fi

if [ ! -f ./vocab.json ]; then
  curl -SL -O https://hf-mirror.com/t-tech/T-one/resolve/main/vocab.json
fi

if [ ! -f ./russian_test_short_from_t_one.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/russian_test_short_from_t_one.wav
fi

python3 ./add_meta_data.py

if [ ! -f ./tokens.txt ]; then
  python3 ./generate_tokens.py
fi

./test.py  --model ./model.onnx  --tokens ./tokens.txt --wave ./russian_test_short_from_t_one.wav
