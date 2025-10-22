#!/usr/bin/env bash

set -ex

cd dotnet-examples/

cd ./version-test
./run.sh
ls -lh

cd ../offline-audio-tagging
./run.sh
ls -lh
rm -rf sherpa-onnx-*

cd ../kitten-tts
./run-kitten.sh
ls -lh
rm -rf kitten-nano-en-v0_1-fp16

cd ../vad-non-streaming-asr-paraformer
./run-ten-vad.sh
rm -fv *.onnx

./run.sh
rm -fv *.onnx

cd ../non-streaming-canary-decode-files
./run.sh
ls -lh
rm -rf sherpa-onnx-nemo-*

cd ../offline-decode-files

./run-wenet-ctc.sh
rm -rf sherpa-onnx-*

./run-zipformer-ctc.sh
rm -rf sherpa-onnx-*

./run-dolphin-ctc.sh
rm -rf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02

./run-fire-red-asr.sh
rm -rf sherpa-onnx-fire-red-asr-*

./run-moonshine.sh
rm -rf sherpa-onnx-*

./run-sense-voice-ctc.sh
rm -rf sherpa-onnx-*

./run-paraformer-itn.sh
rm -rf sherpa-onnx-*

./run-telespeech-ctc.sh
rm -rf sherpa-onnx-*

./run-nemo-ctc.sh
rm -rf sherpa-onnx-*

./run-paraformer.sh
rm -rf sherpa-onnx-*

./run-zipformer.sh
rm -rf sherpa-onnx-*

./run-hotwords.sh
rm -rf sherpa-onnx-*

./run-whisper.sh
rm -rf sherpa-onnx-*

# ./run-whisper-large-v3.sh
# rm -rf sherpa-onnx-*

./run-tdnn-yesno.sh
rm -rf sherpa-onnx-*

cd ../speech-enhancement-gtcrn
./run.sh
ls -lh

cd ../kokoro-tts
./run-kokoro.sh
ls -lh

cd ../offline-tts
./run-matcha-zh.sh
ls -lh *.wav
./run-matcha-en.sh
ls -lh *.wav
./run-aishell3.sh
ls -lh *.wav
./run-piper.sh
ls -lh *.wav
./run-hf-fanchen.sh
ls -lh *.wav
ls -lh

pushd ../..

mkdir tts

cp -v dotnet-examples/kokoro-tts/*.wav ./tts
cp -v dotnet-examples/offline-tts/*.wav ./tts
popd

cd ../offline-speaker-diarization
./run.sh
rm -rfv *.onnx
rm -fv *.wav
rm -rfv sherpa-onnx-pyannote-*

cd ../keyword-spotting-from-files
./run.sh

cd ../online-decode-files
./run-t-one-ctc.sh
rm -rf sherpa-onnx-*

./run-transducer-itn.sh
rm -rf sherpa-onnx-*

./run-zipformer2-ctc.sh
rm -rf sherpa-onnx-*

./run-transducer.sh
rm -rf sherpa-onnx-*

./run-paraformer.sh
rm -rf sherpa-onnx-*

cd ../offline-punctuation
./run.sh
rm -rf sherpa-onnx-*

cd ../speaker-identification
./run.sh

cd ../streaming-hlg-decoding/
./run.sh
rm -rf sherpa-onnx-*

cd ../spoken-language-identification
./run.sh
rm -rf sherpa-onnx-*
