#!/usr/bin/env bash

set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd rust-api-examples

trap 'bash ../.github/scripts/show-rust-binary-info.sh --all || true' EXIT

bash ./run-version.sh

bash ./run-qwen3-asr.sh
rm -rf sherpa-onnx-qwen3-*

bash ./run-funasr-nano.sh
rm -rf sherpa-onnx-funasr-nano-*

bash ./run-audio-tagging-zipformer.sh
rm -rf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15

bash ./run-audio-tagging-ced.sh
rm -rf sherpa-onnx-ced-mini-audio-tagging-2024-04-19

bash ./run-speaker-embedding-extractor.sh
bash ./run-speaker-embedding-manager.sh
rm -f 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
rm -rf sr-data

bash ./run-speaker-embedding-cosine-similarity.sh
rm -f wespeaker_zh_cnceleb_resnet34.onnx fangjun-sr-1.wav fangjun-sr-2.wav leijun-sr-1.wav

bash ./run-offline-speaker-diarization.sh
rm -rf sherpa-onnx-pyannote-segmentation-3-0
rm -f 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx 0-four-speakers-zh.wav

bash ./run-vits-en.sh
rm -rf vits-piper-en_US-amy-low

bash ./run-vits-de.sh
rm -rf vits-piper-de_DE-glados-high

bash ./run-matcha-tts-en.sh
bash ./run-matcha-tts-zh.sh
rm -rf matcha-icefall-en_US-ljspeech matcha-icefall-zh-baker
rm -f vocos-22khz-univ.onnx

bash ./run-kokoro-tts-en.sh
rm -rf kokoro-en-v0_19

bash ./run-kokoro-tts-zh-en.sh
rm -rf kokoro-multi-lang-v1_0

bash ./run-kitten-tts-en.sh
rm -rf kitten-nano-en-v0_1-fp16

bash ./run-pocket-tts.sh
rm -rf sherpa-onnx-pocket-*

bash ./run-supertonic-tts.sh
rm -rf sherpa-onnx-supertonic-*

bash ./run-zipvoice-tts.sh
rm -rf sherpa-onnx-zipvoice-*
rm -f vocos_24khz.onnx

bash ./run-online-punctuation.sh
rm -rf sherpa-onnx-online-punct-*

bash ./run-keyword-spotter.sh
rm -rf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01-mobile

bash ./run-spoken-language-identification.sh
bash ./run-whisper.sh
rm -rf sherpa-onnx-whisper-tiny spoken-language-identification-test-wavs

bash ./run-offline-punctuation.sh
rm -rf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8

bash ./run-moonshine-v2.sh

bash ./run-fire-red-asr-ctc.sh

bash ./run-paraformer.sh
rm -rf sherpa-onnx-paraformer-zh-small-2024-03-09

bash ./run-silero-vad-remove-silence.sh

bash ./run-ten-vad-remove-silence.sh

bash ./run-nemo-parakeet-en.sh
bash ./run-zipformer-vi.sh
bash ./run-zipformer-zh-en.sh
bash ./run-zipformer-en.sh

bash ./run-sense-voice.sh

bash ./run-streaming-zipformer-en.sh
bash ./run-streaming-zipformer-zh-en.sh

bash ./run-offline-speech-enhancement-gtcrn.sh
bash ./run-offline-speech-enhancement-dpdfnet.sh
bash ./run-streaming-speech-enhancement-gtcrn.sh
bash ./run-streaming-speech-enhancement-dpdfnet.sh

bash ./run-cohere-transcribe.sh
rm -rf sherpa-onnx-cohere-transcribe-*
