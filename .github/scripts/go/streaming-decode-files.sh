#!/usr/bin/env bash
set -ex
cd go-api-examples/streaming-decode-files
go mod tidy
go build
./run-paraformer.sh
./run-t-one-ctc.sh
./run-transducer-itn.sh
./run-transducer.sh
./run-zipformer2-ctc-with-hr.sh
./run-zipformer2-ctc.sh
