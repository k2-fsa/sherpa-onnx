#!/usr/bin/env bash
set -ex
cd go-api-examples/speaker-identification
go mod tidy
go build
./run.sh
