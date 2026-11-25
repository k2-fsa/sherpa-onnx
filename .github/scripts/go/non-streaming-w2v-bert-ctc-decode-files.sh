#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-w2v-bert-ctc-decode-files
go mod tidy
go build
./run.sh
