#!/usr/bin/env bash
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)

export SHERPA_ONNX_IS_REVAI=1

set -ex
function install_pyannote() {
  pip install pyannote.audio onnx onnxruntime
}

function download_test_files() {
  curl -SL -O https://huggingface.co/Revai/reverb-diarization-v1/resolve/main/pytorch_model.bin
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

curl -SL -O https://huggingface.co/Revai/reverb-diarization-v1/resolve/main/LICENSE

cat >README.md << EOF
# Introduction

Models in this file are converted from
https://huggingface.co/Revai/reverb-diarization-v1/tree/main

Note that it is accessible under a non-commercial license.

Please see ./LICENSE for details.

See also
https://www.rev.com/blog/speech-to-text-technology/introducing-reverb-open-source-asr-diarization

EOF


