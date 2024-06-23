#!/usr/bin/env bash

set -ex

echo "pwd: $PWD"

cd swift-api-examples
ls -lh

./run-keyword-spotting-from-file.sh
rm ./keyword-spotting-from-file
rm -rf sherpa-onnx-kws-*

./run-streaming-hlg-decode-file.sh
rm ./streaming-hlg-decode-file
rm -rf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

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
