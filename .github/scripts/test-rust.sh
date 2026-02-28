#!/usr/bin/env bash

set -ex

cd rust-api-examples

./run-version.sh

./run-moonshine-v2.sh

./run-fire-red-asr-ctc.sh

./run-silero-vad-remove-silence.sh

./run-nemo-parakeet-en.sh
./run-zipformer-vi.sh
./run-zipformer-zh-en.sh
./run-zipformer-en.sh

./run-sense-voice.sh

./run-streaming-zipformer-en.sh
./run-streaming-zipformer-zh-en.sh
