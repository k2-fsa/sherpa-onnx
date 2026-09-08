#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-tts
go mod tidy
go build
./run-kitten-en.sh
./run-kokoro-en.sh
./run-kokoro-zh-en.sh
./run-matcha-en.sh
./run-matcha-zh.sh
./run-vits-ljs.sh
./run-vits-piper-en_US-lessac-medium.sh
./run-vits-vctk.sh
./run-vits-zh-aishell3.sh
