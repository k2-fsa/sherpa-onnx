#!/usr/bin/env bash
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)
#
set -ex

if [ ! -f kokoro-v1_1-zh.pth ]; then
  curl -SL -O https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth
fi


if [ ! -f config.json ]; then
  # see https://huggingface.co/hexgrad/Kokoro-82M/blob/main/config.json
  curl -SL -O https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json
fi

voices=(
af_maple
af_sol
bf_vale
)
# zf_001-zf_099
for i in $(seq 1 99); do
  a=$(printf "zf_%03d" $i)
  voices+=($a)
done

# zm_009-zm_100
for i in $(seq 9 100); do
  a=$(printf "zm_%03d" $i)
  voices+=($a)
done

echo ${voices[@]} # all elements
echo ${#voices[@]} # length

mkdir -p voices

for v in ${voices[@]}; do
  if [ ! -f voices/$v.pt ]; then
    curl -SL --output voices/$v.pt https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/voices/$v.pt
  fi
done
pushd voices
find . -type f -size -10k -exec rm -v {} +
ls -lh
du -h -d1 .
popd

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
