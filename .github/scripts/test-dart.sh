#!/usr/bin/env bash

set -ex

cd dart-api-examples

pushd non-streaming-asr
echo '----------paraformer----------'
./run-paraformer.sh

echo '----------whisper----------'
./run-whisper.sh

echo '----------zipformer transducer----------'
./run-zipformer-transducer.sh

popd

pushd vad
./run.sh
popd

