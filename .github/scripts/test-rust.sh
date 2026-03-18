#!/usr/bin/env bash

set -ex

cd rust-api-examples

./run-pocket-tts.sh
rm -rf sherpa-onnx-pocket-*

./run-supertonic-tts.sh
rm -rf sherpa-onnx-supertonic-*

./run-online-punctuation.sh
rm -rf sherpa-onnx-online-punct-*

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

./run-offline-speech-enhancement-gtcrn.sh
./run-offline-speech-enhancement-dpdfnet.sh
./run-streaming-speech-enhancement-gtcrn.sh
./run-streaming-speech-enhancement-dpdfnet.sh
