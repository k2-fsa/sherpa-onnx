#!/usr/bin/env bash

set -ex

echo "pwd: $PWD"

cd swift-api-examples
ls -lh

./run-spoken-language-identification.sh
rm -rf sherpa-onnx-whisper*

mkdir -p /Users/fangjun/Desktop
pushd /Users/fangjun/Desktop
curl -SL -O https://huggingface.co/csukuangfj/test-data/resolve/main/Obama.wav
ls -lh
popd

./run-generate-subtitles.sh

ls -lh /Users/fangjun/Desktop
cat /Users/fangjun/Desktop/Obama.srt

./run-tts.sh
ls -lh

./run-decode-file.sh
rm decode-file
sed -i.bak  '20d' ./decode-file.swift
./run-decode-file.sh

./run-decode-file-non-streaming.sh

ls -lh
