#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-moonshine-v2-decode-files
go mod tidy
go build
./run.sh
