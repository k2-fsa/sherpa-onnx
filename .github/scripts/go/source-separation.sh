#!/usr/bin/env bash
set -ex
cd go-api-examples/source-separation
go mod tidy
go build
./run-spleeter.sh
./run-uvr.sh
