#!/usr/bin/env bash
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

pip install \
  nemo_toolkit['asr'] \
  "numpy<2" \
  ipython \
  kaldi-native-fbank \
  librosa \
  onnx \
  onnxruntime \
  onnxscript \
  soundfile

num_frames=1000
if [ $# -ge 1 ]; then
  num_frames="$1"
fi


model_id=nvidia/parakeet-tdt_ctc-110m
if [ $# -ge 2 ]; then
  model_id="$2"
fi

python3 ./wrapper.py --max-len $num_frames --model-id $model_id
