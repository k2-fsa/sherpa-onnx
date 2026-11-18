#!/usr/bin/env bash
set -ex

if [ ! -f ./sherpa-onnx-nemotron-3-diarization/model.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-nemotron-3-diarization.tar.bz2
  tar xvf sherpa-onnx-nemotron-3-diarization.tar.bz2
  rm sherpa-onnx-nemotron-3-diarization.tar.bz2
fi

if [ ! -f ./0-four-speakers-zh.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav
fi

cargo run --example offline_speaker_diarization_sortformer
