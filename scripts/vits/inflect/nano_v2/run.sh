#!/usr/bin/env bash
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

[[ -d ./Inflect-Nano-v2 ]] || git clone https://huggingface.co/owensong/Inflect-Nano-v2

pushd Inflect-Nano-v2
pip install -r ./requirements.txt
popd

export PYTHONPATH=$PWD/Inflect-Nano-v2/runtime:$PYTHONPATH
python3 ./export_onnx.py Inflect-Nano-v2

python3 ./test.py

ls -lh
