#!/usr/bin/env bash

set -ex

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
model=sherpa-onnx-medasr-ctc-en-int8-2025-12-25
archive=$model.tar.bz2

model_dir="$script_dir/$model"
archive_path="$script_dir/$archive"

if [ ! -f "$model_dir/model.int8.onnx" ] ||
   [ ! -f "$model_dir/tokens.txt" ] ||
   [ ! -f "$model_dir/test_wavs/0.wav" ]; then
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$archive_path" "$repo_url/$archive"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$archive_path" "$repo_url/$archive"
  else
    echo "Error: Neither curl nor wget found. Please install one of them."
    exit 1
  fi

  tar xvf "$archive_path" -C "$script_dir"
  rm "$archive_path"
fi

cargo run --example medasr_ctc -- \
  --model "$model_dir/model.int8.onnx" \
  --tokens "$model_dir/tokens.txt" \
  --wav "$model_dir/test_wavs/0.wav" \
  --provider cpu
