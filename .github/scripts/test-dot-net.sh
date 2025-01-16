#!/usr/bin/env bash

cd dotnet-examples/

cd ./kokoro-tts
./run-kokoro-en.sh
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

cd ../offline-decode-files
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

./run-whisper-large-v3.sh
rm -rf sherpa-onnx-*

./run-tdnn-yesno.sh
rm -rf sherpa-onnx-*

cd ../keyword-spotting-from-files
./run.sh

cd ../online-decode-files
./run-transducer-itn.sh
rm -rf sherpa-onnx-*

./run-zipformer2-ctc.sh
rm -rf sherpa-onnx-*

./run-transducer.sh
rm -rf sherpa-onnx-*

./run-paraformer.sh
rm -rf sherpa-onnx-*

cd ../vad-non-streaming-asr-paraformer
./run.sh

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


