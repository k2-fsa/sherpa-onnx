#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-speaker-diarization
go mod tidy
go build
./run.sh

# Uncomment after sherpa-onnx-nemotron-3-diarization.tar.bz2 is published
# cd ../non-streaming-speaker-diarization-sortformer
# go mod tidy
# go build
# ./run.sh
