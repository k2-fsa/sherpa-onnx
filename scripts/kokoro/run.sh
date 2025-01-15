#!/usr/bin/env bash
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

cat > README-new.md <<EOF
# Introduction

Files in this folder are from
https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files

Please see also
https://huggingface.co/hexgrad/Kokoro-82M
and
https://huggingface.co/hexgrad/Kokoro-82M/discussions/14
EOF

files=(
kokoro-v0_19_hf.onnx
# kokoro-v0_19.onnx
# kokoro-quant.onnx
# kokoro-quant-convinteger.onnx
voices.json
)

for f in ${files[@]}; do
  if [ ! -f ./$f ]; then
    curl -SL -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/$f
  fi
done

models=(
# kokoro-v0_19
# kokoro-quant
# kokoro-quant-convinteger
kokoro-v0_19_hf
)

for m in ${models[@]}; do
  ./add-meta-data.py --model $m.onnx --voices ./voices.json
done

ls -l
echo "----------"
ls -lh

for m in ${models[@]}; do
  ./test.py --model $m.onnx --voices-bin ./voices.bin --tokens ./tokens.txt
done
ls -lh
