#!/usr/bin/env bash

if [ -z $HF_TOKEN ]; then
  echo "Please first run export HF_TOKEN=your_huggingface_access_token."
  exit 1
fi

pip install \
  accelerate \
  bitsandbytes \
  git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5 \
  onnx==1.17.0 \
  onnxruntime==1.17.1 \
  librosa \
  onnxscript \
  "numpy<2"

./export_onnx.py

ls -lh
