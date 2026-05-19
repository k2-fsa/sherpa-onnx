// cxx-api-examples/offline-speaker-diarization-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to use offline speaker diarization
// with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
// tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
// rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav
//
// clang-format on

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  OfflineSpeakerDiarizationConfig config;
  config.segmentation.pyannote.model =
      "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx";
  config.embedding.model =
      "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";
  config.clustering.num_clusters = 4;

  OfflineSpeakerDiarization sd = OfflineSpeakerDiarization::Create(config);
  if (!sd.Get()) {
    std::cerr << "Failed to create speaker diarization\n";
    return -1;
  }

  Wave wave = ReadWave("./0-four-speakers-zh.wav");
  if (wave.samples.empty()) {
    std::cerr << "Failed to read wave file\n";
    return -1;
  }

  std::vector<OfflineSpeakerDiarizationSegment> segments =
      sd.Process(wave.samples.data(), wave.samples.size());

  for (const auto &seg : segments) {
    printf("%.3f -- %.3f speaker_%02d\n", seg.start, seg.end, seg.speaker);
  }

  return 0;
}
