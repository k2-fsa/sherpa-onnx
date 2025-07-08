#!/usr/bin/env bash
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

cat > README-new.md <<EOF
# Introduction

Files in this folder are from
git clone https://huggingface.co/hexgrad/kLegacy
EOF

if [ ! -d kLegacy ]; then
  git clone https://huggingface.co/hexgrad/kLegacy
  pushd kLegacy/v0.19
  git lfs pull
  popd
fi

if [ ! -f ./voices.bin ]; then
  ./generate_voices_bin.py
fi

if [ ! -f ./tokens.txt ]; then
  ./generate_tokens.py
fi

if [ ! -f ./model.onnx ]; then
  mv kLegacy/v0.19/kokoro-v0_19.onnx ./model.onnx
fi

./add_meta_data.py --model ./model.onnx

if [ ! -f model.int8.onnx ]; then
  ./dynamic_quantization.py
fi
