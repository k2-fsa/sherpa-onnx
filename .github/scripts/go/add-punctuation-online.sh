#!/usr/bin/env bash
set -ex
cd go-api-examples/add-punctuation-online
go mod tidy
go build
./run.sh
