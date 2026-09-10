#!/usr/bin/env bash
set -ex
cd go-api-examples/keyword-spotting-from-file
go mod tidy
go build
./run.sh
