#!/usr/bin/env bash

set -ex

cd rust-api-examples

./run-vits-tts.sh
rm -rf vits-piper-en_US-amy-low

./run-vits-tts-de.sh
rm -rf vits-piper-de_DE-glados-high

./run-matcha-tts-en.sh
./run-matcha-tts-zh.sh
rm -rf matcha-icefall-en_US-ljspeech matcha-icefall-zh-baker
rm -f vocos-22khz-univ.onnx

./run-kokoro-tts-en.sh
rm -rf kokoro-en-v0_19

./run-kokoro-tts-zh-en.sh
rm -rf kokoro-multi-lang-v1_0

./run-kitten-tts-en.sh
rm -rf kitten-nano-en-v0_1-fp16

./run-pocket-tts.sh
rm -rf sherpa-onnx-pocket-*

./run-supertonic-tts.sh
rm -rf sherpa-onnx-supertonic-*

./run-zipvoice-tts.sh
rm -rf sherpa-onnx-zipvoice-*
rm -f vocos_24khz.onnx

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
