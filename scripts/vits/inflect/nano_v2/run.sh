#!/usr/bin/env bash

set -ex

[[ -f ./decode.onnx ]] || wget https://huggingface.co/owensong/Inflect-Nano-v2-ONNX/resolve/main/onnx/decode.onnx
[[ -f ./duration.onnx ]] ||wget https://huggingface.co/owensong/Inflect-Nano-v2-ONNX/resolve/main/onnx/duration.onnx
