#!/usr/bin/env bash

set -ex

dart pub get

if [ ! -f ./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
fi

if [ ! -f ./sr-data/enroll/leijun-sr-1.wav ]; then
  curl -SL -o sr-data.tar.gz https://github.com/csukuangfj/sr-data/archive/refs/tags/v1.0.0.tar.gz
  tar xvf sr-data.tar.gz
  mv sr-data-1.0.0 sr-data
fi

dart run \
  ./bin/speaker_id.dart \
  --model ./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
