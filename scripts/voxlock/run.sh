#!/usr/bin/env bash
# Copyright    2026  Arbaz Siddiqui

set -ex

# Weights of VoxLock 1.1.1
revision=a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03

models=(
enroll.onnx
chunk.onnx
chunk_int8.onnx
chunk_int8_full.onnx
enroll_fp16.onnx
chunk_fp16.onnx
)

mkdir -p onnx
for m in ${models[@]}; do
  if [ ! -f onnx/$m ]; then
    curl -SL --output onnx/$m https://huggingface.co/arbazsiddiqui/voxlock/resolve/$revision/onnx/$m
  fi
done

python3 ./add_meta_data.py --in-dir ./onnx --out-dir ./models

ls -lh models
