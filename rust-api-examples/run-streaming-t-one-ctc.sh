#!/usr/bin/env bash

set -ex

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# see
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/t-one.html
repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
model=sherpa-onnx-streaming-t-one-russian-2025-09-08
archive=$model.tar.bz2

model_dir="$script_dir/$model"
archive_path="$script_dir/$archive"

if [ ! -f "$model_dir/model.onnx" ] ||
   [ ! -f "$model_dir/tokens.txt" ] ||
   [ ! -f "$model_dir/0.wav" ]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$archive_path" "$repo_url/$archive"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$archive_path" "$repo_url/$archive"
  else
    echo "Error: Neither curl nor wget found. Please install one of them."
    exit 1
  fi

  tar xvf "$archive_path" -C "$script_dir"
  rm "$archive_path"
fi

cargo run --example streaming_t_one_ctc -- \
  --model "$model_dir/model.onnx" \
  --tokens "$model_dir/tokens.txt" \
  --wav "$model_dir/0.wav" \
  --provider cpu
