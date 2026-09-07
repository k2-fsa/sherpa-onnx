#!/usr/bin/env bash
set -ex
cd go-api-examples/streaming-decode-files
go mod tidy
. "$(dirname "${BASH_SOURCE[0]}")/replace-sherpa-onnx-go.sh"
go build
./run-paraformer.sh
./run-t-one-ctc.sh
./run-transducer-itn.sh
./run-transducer.sh
./run-zipformer2-ctc-with-hr.sh
./run-zipformer2-ctc.sh
