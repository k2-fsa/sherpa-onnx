#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-tts-phonemize
go mod tidy
go build
./run-vits-piper-phonemize.sh
./run-kitten-en-phonemize.sh
./run-kokoro-en-phonemize.sh
./run-matcha-en-phonemize.sh
