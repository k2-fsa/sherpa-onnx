#!/usr/bin/env bash
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

pip install \
  "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main" \
  "numpy<2" \
  ipython \
  kaldi-native-fbank \
  librosa \
  onnx \
  onnxruntime \
  onnxscript \
  soundfile

# 80, 160, 320, 560, 1120
chunk_size_ms=1120
if [ $# -ge 1 ]; then
  chunk_size_ms="$1"
fi


model_id=nemotron-3.5-asr-streaming-0.6b
if [ $# -ge 2 ]; then
  model_id="$2"
fi

python3 ./wrapper.py --chunk-size-ms $chunk_size_ms --model-id $model_id
