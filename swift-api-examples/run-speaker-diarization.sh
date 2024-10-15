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

if [ ! -e ./speaker-diarization ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/install/include \
    -import-objc-header ./SherpaOnnx-Bridging-Header.h \
    ./speaker-diarization.swift  ./SherpaOnnx.swift \
    -L ../build-swift-macos/install/lib/ \
    -l sherpa-onnx \
    -l onnxruntime \
    -o speaker-diarization

  strip speaker-diarization
else
  echo "./speaker-diarization exists - skip building"
fi

export DYLD_LIBRARY_PATH=$PWD/../build-swift-macos/install/lib:$DYLD_LIBRARY_PATH
./speaker-diarization
