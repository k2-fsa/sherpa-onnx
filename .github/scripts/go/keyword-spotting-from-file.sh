#!/usr/bin/env bash
set -ex
cd go-api-examples/keyword-spotting-from-file
go mod tidy
. "$(dirname "${BASH_SOURCE[0]}")/replace-sherpa-onnx-go.sh"
go build
./run.sh
