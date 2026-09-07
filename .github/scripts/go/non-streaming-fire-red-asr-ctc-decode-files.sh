#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-fire-red-asr-ctc-decode-files
go mod tidy
. "$(dirname "$0")/replace-sherpa-onnx-go.sh"
go build
./run.sh
