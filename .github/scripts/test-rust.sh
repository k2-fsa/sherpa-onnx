#!/usr/bin/env bash

set -ex

cd rust-api-examples

./run-version.sh

./run-streaming-zipformer-en.sh
./run-streaming-zipformer-zh-en.sh
