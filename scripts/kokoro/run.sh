#!/usr/bin/env bash
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex


files=(
kokoro-v0_19_hf.onnx
kokoro-v0_19.onnx
kokoro-quant.onnx
kokoro-quant-convinteger.onnx
voices.json
)

for f in ${files[@]}; do
  if [ ! -f ./$f ]; then
    curl -SL -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/$f
  fi
done

models=(
kokoro-v0_19_hf
kokoro-v0_19
kokoro-quant
kokoro-quant-convinteger
)

for m in ${models[@]}; do
  ./add-meta-data.py --model $m.onnx --voices ./voices.json
done

for m in ${models[@]}; do
  ./test.py --model $m.onnx --voices-bin ./voices.json --tokens ./tokens.txt
done
ls -lh
