#!/usr/bin/env bash
set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd go-api-examples/streaming-decode-files
go mod tidy
. "$SCRIPT_DIR/replace-sherpa-onnx-go.sh"
go build
./run-paraformer.sh
./run-t-one-ctc.sh
./run-transducer-itn.sh
./run-transducer.sh
./run-zipformer2-ctc-with-hr.sh
./run-zipformer2-ctc.sh
