#!/usr/bin/env bash

set -ex

cd rust-api-examples

./run-version.sh

./run-streaming-zipformer.sh
