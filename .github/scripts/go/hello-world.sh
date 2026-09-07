#!/usr/bin/env bash
set -ex
cd go-api-examples/hello-world
go mod tidy
go build
./run.sh
