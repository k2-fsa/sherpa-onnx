#!/usr/bin/env bash

if [ ! -f ./3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
fi

if [ ! -f ./sr-data/enroll/fangjun-sr-1.wav ]; then
  git clone https://github.com/csukuangfj/sr-data
fi

if [ ! -f ./silero_vad.onnx ]; then
  curl -SL -O https://github.com/snakers4/silero-vad/blob/master/files/silero_vad.onnx
fi

go mod tidy
go build
./vad-speaker-identification
