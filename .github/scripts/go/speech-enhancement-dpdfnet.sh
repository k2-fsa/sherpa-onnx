#!/usr/bin/env bash
set -ex
cd go-api-examples/speech-enhancement-dpdfnet
go mod tidy
go build
./run.sh
