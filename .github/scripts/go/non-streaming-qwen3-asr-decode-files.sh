#!/usr/bin/env bash
set -ex
cd go-api-examples/non-streaming-qwen3-asr-decode-files
go mod tidy
. "$(dirname "${BASH_SOURCE[0]}")/replace-sherpa-onnx-go.sh"
go build
./run.sh
