#!/usr/bin/env bash
set -ex
cd go-api-examples/audio-tagging
go mod tidy
. "$(dirname "$0")/replace-sherpa-onnx-go.sh"
go build
./run.sh
