#!/usr/bin/env bash
set -ex
cd go-api-examples/source-separation
go mod tidy
. "$(dirname "$0")/replace-sherpa-onnx-go.sh"
go build
./run-spleeter.sh
./run-uvr.sh
