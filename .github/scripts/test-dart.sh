#!/usr/bin/env bash

set -ex

cd dart-api-examples

pushd non-streaming-asr
./run-whisper.sh
popd

pushd vad
./run.sh
popd

