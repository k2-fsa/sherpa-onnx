#!/usr/bin/env bash
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex
function install_pyannote() {
  pip install pyannote.audio onnx onnxruntime
}

function download_test_files() {
  curl -SL -O https://huggingface.co/csukuangfj/pyannote-models/resolve/main/segmentation-3.0/pytorch_model.bin
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
}

install_pyannote
download_test_files

./export-onnx.py
./preprocess.sh

echo "----------torch----------"
./vad-torch.py

echo "----------onnx model.onnx----------"
./vad-onnx.py --model ./model.onnx --wav ./lei-jun-test.wav

echo "----------onnx model.int8.onnx----------"
./vad-onnx.py --model ./model.int8.onnx --wav ./lei-jun-test.wav

cat >README.md << EOF
# Introduction

Models in this file are converted from
https://huggingface.co/pyannote/segmentation-3.0/tree/main

EOF

cat >LICENSE <<EOF
MIT License

Copyright (c) 2022 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
