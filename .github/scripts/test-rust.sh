#!/usr/bin/env bash

set -ex

cd rust-api-examples

./run-version.sh

./run-nemo-parakeet-en.sh
./run-zipformer-vi.sh
./run-zipformer-zh-en.sh
./run-zipformer-en.sh

./run-sense-voice.sh

./run-streaming-zipformer-en.sh
./run-streaming-zipformer-zh-en.sh
