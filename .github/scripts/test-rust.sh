#!/usr/bin/env bash

set -ex

cd rust-api-examples

trap 'bash ../.github/scripts/show-rust-binary-info.sh --all || true' EXIT

./run-cohere-transcribe.sh
rm -rf sherpa-onnx-cohere-transcribe-*

./run-qwen3-asr.sh
rm -rf sherpa-onnx-qwen3-*

./run-audio-tagging-zipformer.sh
rm -rf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15

./run-audio-tagging-ced.sh
rm -rf sherpa-onnx-ced-mini-audio-tagging-2024-04-19

./run-speaker-embedding-extractor.sh
./run-speaker-embedding-manager.sh
rm -f 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
rm -rf sr-data

./run-speaker-embedding-cosine-similarity.sh
rm -f wespeaker_zh_cnceleb_resnet34.onnx fangjun-sr-1.wav fangjun-sr-2.wav leijun-sr-1.wav

./run-offline-speaker-diarization.sh
rm -rf sherpa-onnx-pyannote-segmentation-3-0
rm -f 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx 0-four-speakers-zh.wav

./run-vits-en.sh
rm -rf vits-piper-en_US-amy-low

./run-vits-de.sh
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

./run-keyword-spotter.sh
rm -rf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile

./run-spoken-language-identification.sh
rm -rf sherpa-onnx-whisper-tiny spoken-language-identification-test-wavs

./run-offline-punctuation.sh
rm -rf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8

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
