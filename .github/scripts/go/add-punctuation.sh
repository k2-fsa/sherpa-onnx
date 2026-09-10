#!/usr/bin/env bash
set -ex
cd go-api-examples/add-punctuation
go mod tidy
go build
./run.sh
