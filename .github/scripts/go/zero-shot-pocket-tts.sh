#!/usr/bin/env bash
set -ex
cd go-api-examples/zero-shot-pocket-tts
go mod tidy
go build
./run.sh
