#!/usr/bin/env bash
set -ex
cd go-api-examples/streaming-speech-enhancement-voxlock-with-voice-sample
go mod tidy
go build
./run.sh
