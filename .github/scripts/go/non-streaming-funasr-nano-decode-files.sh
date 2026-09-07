#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-funasr-nano-decode-files
go mod tidy
. "$(dirname "$0")/replace-sherpa-onnx-go.sh"
go build
./run.sh
