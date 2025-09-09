#!/usr/bin/env bash
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

if [ ! -f kitten_tts_mini_v0_1.onnx ]; then
  curl -SL -O https://huggingface.co/KittenML/kitten-tts-mini-0.1/resolve/main/kitten_tts_mini_v0_1.onnx
fi

if [ ! -f voices.npz ]; then
  curl -SL -O https://huggingface.co/KittenML/kitten-tts-mini-0.1/resolve/main/voices.npz
fi

./generate_voices_bin.py
./generate_tokens.py

./convert_opset.py
./show.py
./add_meta_data.py --model ./model.fp16.onnx
# ./test.py --model ./model.fp16.onnx --tokens ./tokens.txt --voice ./voices.bin
ls -lh
