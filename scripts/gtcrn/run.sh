#!/usr/bin/env bash
#

if [ ! -f gtcrn_simple.onnx ]; then
  wget https://github.com/Xiaobin-Rong/gtcrn/raw/refs/heads/main/stream/onnx_models/gtcrn_simple.onnx
fi

if [ ! -f ./inp_16k.wav ]; then
  wget https://github.com/yuyun2000/SpeechDenoiser/raw/refs/heads/main/16k/inp_16k.wav
fi

python3 ./add_meta_data.py
