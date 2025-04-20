#!/usr/bin/env bash
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

git clone https://huggingface.co/hexgrad/Kokoro-82M

# https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices
#
# af -> American female
# am -> American male
# bf -> British female
# bm -> British male

if [ ! -f ./kokoro.onnx ]; then
  python3 ./export_onnx.py
fi


if [ ! -f ./.add-meta-data.done ]; then
  python3 ./add_meta_data.py
  touch ./.add-meta-data.done
fi

if [ ! -f ./kokoro.int8.onnx ]; then
  python3 ./dynamic_quantization.py
fi

if [ ! -f us_gold.json ]; then
  curl -SL -O https://raw.githubusercontent.com/hexgrad/misaki/refs/heads/main/misaki/data/us_gold.json
fi

if [ ! -f us_silver.json ]; then
  curl -SL -O https://raw.githubusercontent.com/hexgrad/misaki/refs/heads/main/misaki/data/us_silver.json
fi

if [ ! -f gb_gold.json ]; then
  curl -SL -O https://raw.githubusercontent.com/hexgrad/misaki/refs/heads/main/misaki/data/gb_gold.json
fi

if [ ! -f gb_silver.json ]; then
  curl -SL -O https://raw.githubusercontent.com/hexgrad/misaki/refs/heads/main/misaki/data/gb_silver.json
fi

if [ ! -f ./tokens.txt ]; then
  ./generate_tokens.py
fi

if [ ! -f ./lexicon-zh.txt ]; then
  ./generate_lexicon_zh.py
fi

if [[ ! -f ./lexicon-us-en.txt || ! -f ./lexicon-gb-en.txt ]]; then
  ./generate_lexicon_en.py
fi

if [ ! -f ./voices.bin ]; then
  ./generate_voices_bin.py
fi

./test.py
ls -lh
