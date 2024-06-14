#!/usr/bin/env bash

set -ex

cd dart-api-examples

pushd non-streaming-asr

echo '----------VAD with paraformer----------'
./run-vad-with-paraformer.sh
rm -rf sherpa-onnx-*

echo '----------NeMo transducer----------'
./run-nemo-transducer.sh
rm -rf sherpa-onnx-*

echo '----------NeMo CTC----------'
./run-nemo-ctc.sh
rm -rf sherpa-onnx-*

echo '----------TeleSpeech CTC----------'
./run-telespeech-ctc.sh
rm -rf sherpa-onnx-*

echo '----------paraformer----------'
./run-paraformer.sh
rm -rf sherpa-onnx-*

echo '----------whisper----------'
./run-whisper.sh
rm -rf sherpa-onnx-*

echo '----------zipformer transducer----------'
./run-zipformer-transducer.sh
rm -rf sherpa-onnx-*

popd

pushd vad
./run.sh
rm *.onnx
popd

