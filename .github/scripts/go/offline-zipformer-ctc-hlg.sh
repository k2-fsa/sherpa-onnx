#!/usr/bin/env bash
set -ex
cd go-api-examples/offline-zipformer-ctc-hlg
go mod tidy
go build
./run.sh
