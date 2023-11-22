#!/usr/bin/env bash

set -ex

echo "pwd: $PWD"

cd swift-api-examples
ls -lh

mkdir -p /Users/fangjun/Desktop
pushd /Users/fangjun/Desktop
wget -q https://huggingface.co/csukuangfj/test-data/resolve/main/Obama.wav
ls -lh
popd

./run-generate-subtitles.sh

ls -lh /Users/fangjun/Desktop
cat /Users/fangjun/Desktop/Obama.srt

./run-tts.sh
ls -lh

./run-decode-file.sh

./run-decode-file-non-streaming.sh

ls -lh
