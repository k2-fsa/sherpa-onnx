#!/usr/bin/env bash

set -ex

if [ ! -f ./3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
fi

if [ ! -f ./sr-data/enroll/fangjun-sr-1.wav ]; then
  git clone https://github.com/csukuangfj/sr-data
fi

go mod tidy
go build
./speaker-identification
