#!/usr/bin/env bash

set -ex

cd rust-api-examples

./run-version.sh

./run-sense-voice.sh

./run-streaming-zipformer-en.sh
./run-streaming-zipformer-zh-en.sh
