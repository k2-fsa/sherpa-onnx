#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-omnilingual-asr-ctc-decode-files
go mod tidy
go build
./run.sh
