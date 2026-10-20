#!/usr/bin/env bash

set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
echo "pwd: $PWD"

cd swift-api-examples
ls -lh

./run-test-version.sh

./run-tts-pocket-en.sh
ls -lh
rm -rf sherpa-onnx-pocket-*

./run-tts-supertonic-en.sh
ls -lh
rm -rf sherpa-onnx-supertonic-*

./run-tts-kitten-en-lexicon.sh
ls -lh
rm -rf kitten-*

./run-tts-vits-lexicon.sh
ls -lh
rm -rf vits-piper-*

./run-tts-kokoro-zh-en-lexicon.sh
ls -lh
rm -rf kokoro-multi-*

./run-tts-kokoro-en-lexicon.sh
ls -lh
rm -rf kokoro-en-*

./run-tts-matcha-zh-lexicon.sh
ls -lh
rm -rf matcha-icefall-*

./run-tts-matcha-en-lexicon.sh
ls -lh
rm -rf matcha-icefall-*

./run-tts-zipvoice-lexicon.sh
ls -lh
rm -rf sherpa-onnx-zipvoice-*
rm -f vocos_24khz.onnx
rm -f lexicon-en-us.txt

./run-cohere-transcribe.sh
rm -rf sherpa-onnx-cohere-transcribe-*

./run-source-separation-spleeter.sh
rm -rf sherpa-onnx-spleeter-*
rm vocals.wav accompaniment.wav

./run-source-separation-uvr.sh
rm -rf UVR-MDX-NET-Voc_FT.onnx
rm uvr-vocals.wav uvr-non-vocals.wav
rm -f qi-feng-le-zh.wav

./run-qwen3-asr.sh
rm -rf sherpa-onnx-qwen3-*

./run-moonshine-v2-asr.sh
rm -rf sherpa-onnx-moonshine-*

./run-fire-red-asr-ctc.sh
rm -rf sherpa-onnx-fire-red-*

./run-medasr-ctc-asr.sh
rm -rf sherpa-onnx-medasr-*

./run-funasr-nano-asr.sh
rm -rf sherpa-onnx-funasr-nano-*

./run-omnilingual-asr-ctc-asr.sh
rm -rf sherpa-onnx-omnilingual-*

./run-decode-file-t-one-streaming.sh
rm -rf sherpa-onnx-streaming-*

./run-compute-speaker-embeddings.sh
rm -f wespeaker_zh_cnceleb_resnet34.onnx fangjun-sr-1.wav fangjun-sr-2.wav leijun-sr-1.wav

./run-wenet-ctc-asr.sh
rm -rf sherpa-onnx-*

./run-zipformer-ctc-asr.sh
rm -rf sherpa-onnx-zipformer-*

./run-zipformer-ctc-hlg.sh
rm -rf sherpa-onnx-zipformer-*

./run-decode-file-sense-voice-with-hr.sh
rm -rf sherpa-onnx-sense-voice-*
rm -rf dict lexicon.txt replace.fst test-hr.wav

./run-dolphin-ctc-asr.sh
rm -rf sherpa-onnx-dolphin-*

./run-speech-enhancement-gtcrn.sh
./run-speech-enhancement-dpdfnet.sh
./run-online-speech-enhancement-gtcrn.sh
./run-online-speech-enhancement-dpdfnet.sh
ls -lh *.wav

./run-fire-red-asr.sh
rm -rf sherpa-onnx-fire-red-asr-*

./run-speaker-diarization.sh
rm -f 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx 0-four-speakers-zh.wav
rm -rf sherpa-onnx-pyannote-segmentation-3-0

./run-add-punctuations.sh
rm ./add-punctuations
rm -rf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12

./run-add-punctuations-online.sh
rm ./add-punctuation-online
rm -rf sherpa-onnx-online-punct-en-2024-08-06

./run-keyword-spotting-from-file.sh
rm ./keyword-spotting-from-file
rm -rf sherpa-onnx-kws-*

./run-streaming-hlg-decode-file.sh
rm ./streaming-hlg-decode-file
rm -rf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

./run-spoken-language-identification.sh
rm -rf sherpa-onnx-whisper*

mkdir -p /Users/fangjun/Desktop
pushd /Users/fangjun/Desktop
download https://huggingface.co/csukuangfj/test-data/resolve/main/Obama.wav
ls -lh
popd

./run-generate-subtitles-ten-vad.sh
rm -rf *.onnx

./run-generate-subtitles.sh
rm -rf *.onnx

ls -lh /Users/fangjun/Desktop
cat /Users/fangjun/Desktop/Obama.srt

rm -rf sherpa-onnx-whisper*
rm -f *.onnx
rm /Users/fangjun/Desktop/Obama.wav

./run-decode-file.sh
rm decode-file
sed -i.bak  '20d' ./decode-file.swift
./run-decode-file.sh

./run-decode-file-non-streaming.sh

ls -lh
