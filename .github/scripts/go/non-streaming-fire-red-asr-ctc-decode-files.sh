#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-fire-red-asr-ctc-decode-files
go mod tidy
go build
./run.sh
