#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-voxlock-int8 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
  tar xvf sherpa-onnx-voxlock-int8.tar.bz2
  rm sherpa-onnx-voxlock-int8.tar.bz2
fi

cargo run --example offline_speech_enhancement_voxlock_with_voice_sample -- \
  --enroll-model ./sherpa-onnx-voxlock-int8/enroll.onnx \
  --chunk-model ./sherpa-onnx-voxlock-int8/chunk_int8.onnx \
  --voice-sample ./sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav \
  --input ./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav \
  --output ./voxlock-enhanced-with-voice-sample-16k.wav

mkdir -p offline_speech_enhancement_voxlock_with_voice_sample
mv voxlock-enhanced-with-voice-sample-16k.wav offline_speech_enhancement_voxlock_with_voice_sample/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav offline_speech_enhancement_voxlock_with_voice_sample/
