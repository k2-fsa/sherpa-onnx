#!/usr/bin/env bash


if [ ! -f ./sherpa-onnx-pyannote-segmentation-3-0/model.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
fi

if [ ! -f ./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
fi

if [ ! -f ./0-four-speakers-zh.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav
fi

go mod tidy
go build
./non-streaming-speaker-diarization
