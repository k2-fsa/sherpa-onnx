#!/usr/bin/env bash
set -ex
cd go-api-examples/zero-shot-zipvoice-tts
go mod tidy
go build
./run.sh
