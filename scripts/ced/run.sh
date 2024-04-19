#!/usr/bin/env bash
#
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

function install_dependencies() {
  pip install -qq torch==2.1.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  pip install -qq onnx onnxruntime==1.17.1

  pip install -r ./requirements.txt
}

git clone https://github.com/RicherMans/CED
pushd CED

install_dependencies

models=(
ced_tiny
ced_mini
ced_small
ced_base
)

for m in ${models[@]}; do
  python3 ./export_onnx.py -m $m
done

ls -lh *.onnx
