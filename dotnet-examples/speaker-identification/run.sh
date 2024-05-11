#!/usr/bin/env bash

set -ex

if [ ! -e ./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
fi

if [ ! -d ./sr-data ]; then
  git clone https://github.com/csukuangfj/sr-data
fi

dotnet run
