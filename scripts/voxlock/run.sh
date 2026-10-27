#!/usr/bin/env bash
# Copyright    2026  Arbaz Siddiqui

set -ex

# Models of VoxLock 1.2.0
revision=2413b292bc2c337bfc19a9802aaf3e011e68e9c4

models=(
enroll.onnx
chunk.onnx
chunk_int8.onnx
chunk_int8_full.onnx
enroll_fp16.onnx
chunk_fp16.onnx
)

onnx_dir=onnx/$revision
mkdir -p $onnx_dir
for m in ${models[@]}; do
  if [ ! -f $onnx_dir/$m ]; then
    curl -fSL --output $onnx_dir/$m.tmp https://huggingface.co/arbazsiddiqui/voxlock/resolve/$revision/onnx/v2/$m
    mv $onnx_dir/$m.tmp $onnx_dir/$m
  fi
done

python3 ./add_meta_data.py --in-dir ./$onnx_dir --out-dir ./models

ls -lh models
