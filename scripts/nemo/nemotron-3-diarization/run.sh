#!/usr/bin/env bash
# Copyright (c)  2026  Silvio Tomatis

set -ex

# Requires torch, transformers (with nemotron3_diarization), onnx,
# onnxruntime, librosa and soundfile. See ./README.md

if [ ! -f ./0-four-speakers-zh.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav
fi

python3 ./export_onnx.py

ls -lh *.onnx

# Compare with the PyTorch model from transformers
python3 ./test_onnx.py \
  --model ./model.onnx \
  --wav ./0-four-speakers-zh.wav \
  --reference nvidia/Nemotron-3-Diarization

python3 ./test_onnx.py \
  --model ./model.int8.onnx \
  --wav ./0-four-speakers-zh.wav
