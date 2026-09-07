#!/usr/bin/env bash
set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd go-api-examples/non-streaming-fire-red-asr-ctc-decode-files
go mod tidy
. "$SCRIPT_DIR/replace-sherpa-onnx-go.sh"
go build
./run.sh
