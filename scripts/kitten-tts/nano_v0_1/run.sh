#!/usr/bin/env bash
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

if [ ! -f kitten_tts_nano_v0_1.onnx ]; then
  curl -SL -O https://huggingface.co/KittenML/kitten-tts-nano-0.1/resolve/main/kitten_tts_nano_v0_1.onnx
fi

if [ ! -f voices.npz ]; then
  curl -SL -O https://huggingface.co/KittenML/kitten-tts-nano-0.1/resolve/main/voices.npz
fi
