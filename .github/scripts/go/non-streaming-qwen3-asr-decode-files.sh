#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-qwen3-asr-decode-files
go mod tidy
go build
./run.sh
