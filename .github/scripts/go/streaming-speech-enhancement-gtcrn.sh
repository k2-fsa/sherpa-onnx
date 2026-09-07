#!/usr/bin/env bash
set -ex
cd go-api-examples/streaming-speech-enhancement-gtcrn
go mod tidy
go build
./run.sh
