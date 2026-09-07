#!/usr/bin/env bash
set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd go-api-examples/zero-shot-zipvoice-tts
go mod tidy
. "$SCRIPT_DIR/replace-sherpa-onnx-go.sh"
go build
./run.sh
