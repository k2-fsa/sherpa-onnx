#!/usr/bin/env bash

set -ex

echo "pwd: $PWD"

cd swift-api-examples
ls -lh

./run-test-version.sh

./run-medasr-ctc-asr.sh
rm -rf sherpa-onnx-medasr-*

./run-funasr-nano-asr.sh
rm -rf sherpa-onnx-funasr-nano-*

./run-omnilingual-asr-ctc-asr.sh
rm -rf sherpa-onnx-omnilingual-*

./run-decode-file-t-one-streaming.sh
rm -rf sherpa-onnx-streaming-*

./run-compute-speaker-embeddings.sh
rm -fv *.wav *.onnx

./run-tts-kitten-en.sh
ls -lh
rm -rf kitten-*

./run-wenet-ctc-asr.sh
rm -rf sherpa-onnx-*

./run-zipformer-ctc-asr.sh
rm -rf sherpa-onnx-zipformer-*

./run-decode-file-sense-voice-with-hr.sh
rm -rf sherpa-onnx-sense-voice-*
rm -rf dict lexicon.txt replace.fst test-hr.wav

./run-dolphin-ctc-asr.sh
rm -rf sherpa-onnx-dolphin-*

./run-speech-enhancement-gtcrn.sh
ls -lh *.wav

./run-fire-red-asr.sh
rm -rf sherpa-onnx-fire-red-asr-*

./run-tts-vits.sh
ls -lh
rm -rf vits-piper-*

./run-tts-kokoro-zh-en.sh
ls -lh
rm -rf kokoro-multi-*

./run-tts-kokoro-en.sh
ls -lh
rm -rf kokoro-en-*

./run-tts-matcha-zh.sh
ls -lh
rm -rf matcha-icefall-*

./run-tts-matcha-en.sh
ls -lh
rm -rf matcha-icefall-*

./run-speaker-diarization.sh
rm -rf *.onnx
rm -rf sherpa-onnx-pyannote-segmentation-3-0
rm -fv *.wav

./run-add-punctuations.sh
rm ./add-punctuations
rm -rf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12

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
curl -SL -O https://huggingface.co/csukuangfj/test-data/resolve/main/Obama.wav
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
