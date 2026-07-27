#!/usr/bin/env bash

set -ex

[[ -d Inflect-Nano-v2 ]] || git clone https://huggingface.co/owensong/Inflect-Nano-v2
pushd Inflect-Nano-v2
pip install -r ./requirements.txt
popd
pip install onnxruntime onnx

