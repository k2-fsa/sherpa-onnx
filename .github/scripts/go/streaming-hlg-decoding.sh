#!/usr/bin/env bash
set -ex
cd go-api-examples/streaming-hlg-decoding
go mod tidy
go build
./run.sh
