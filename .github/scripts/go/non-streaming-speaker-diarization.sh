#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-speaker-diarization
go mod tidy
go build
./run.sh
