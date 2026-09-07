#!/usr/bin/env bash
set -ex
cd go-api-examples/audio-tagging
go mod tidy
go build
./run.sh
