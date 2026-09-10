#!/usr/bin/env bash
set -ex
cd go-api-examples/supertonic-tts
go mod tidy
go build
./run.sh
