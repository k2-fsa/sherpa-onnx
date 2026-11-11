#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -d ./sherpa-onnx-voxlock-int8 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
  tar xvf sherpa-onnx-voxlock-int8.tar.bz2
  rm sherpa-onnx-voxlock-int8.tar.bz2
fi

dart run \
  ./bin/streaming_speech_enhancement_voxlock_with_voice_sample.dart \
  --enroll-model ./sherpa-onnx-voxlock-int8/enroll.onnx \
  --chunk-model ./sherpa-onnx-voxlock-int8/chunk_int8.onnx \
  --voice-sample ./sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav \
  --input-wav ./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav \
  --output-wav ./voxlock-enhanced-online-with-voice-sample-16k.wav

ls -lh *.wav
