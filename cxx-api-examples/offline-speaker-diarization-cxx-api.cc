// cxx-api-examples/offline-speaker-diarization-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to implement speaker diarization with
// sherpa-onnx's C++ API.

// clang-format off
/*
Usage:

Step 1: Download a speaker segmentation model

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

Step 2: Download a speaker embedding extractor model

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

Step 3. Download test wave files

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available test wave files. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Step 4. Run it

  ./offline-speaker-diarization-cxx-api

 */
// clang-format on

#include <cstdint>
#include <cstdio>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

static int32_t ProgressCallback(int32_t num_processed_chunks,
                                int32_t num_total_chunks, void *arg) {
  float progress = 100.0 * num_processed_chunks / num_total_chunks;
  fprintf(stderr, "progress %.2f%%\n", progress);

  // the return value is currently ignored
  return 0;
}

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  // Please see the comments at the start of this file for how to download
  // the .onnx file and .wav files below
  std::string segmentation_model =
      "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx";

  std::string embedding_extractor_model =
      "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";

  std::string wav_filename = "./0-four-speakers-zh.wav";

  Wave wave = ReadWave(wav_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read " << wav_filename << "\n";
    return -1;
  }

  SherpaOnnxOfflineSpeakerDiarizationConfig config;
  memset(&config, 0, sizeof(config));

  config.segmentation.pyannote.model = segmentation_model.c_str();
  config.embedding.model = embedding_extractor_model.c_str();

  // the test wave ./0-four-speakers-zh.wav has 4 speakers, so
  // we set num_clusters to 4
  //
  config.clustering.num_clusters = 4;
  // If you don't know the number of speakers in the test wave file, please
  // use
  // config.clustering.threshold = 0.5; // You need to tune this threshold

  const SherpaOnnxOfflineSpeakerDiarization *sd =
      SherpaOnnxCreateOfflineSpeakerDiarization(&config);

  if (!sd) {
    std::cerr << "Failed to initialize offline speaker diarization\n";
    return -1;
  }

  if (SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(sd) !=
      wave.sample_rate) {
    std::cerr << "Expected sample rate: "
              << SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(sd)
              << ". Actual sample rate from the wave file: "
              << wave.sample_rate << "\n";
    SherpaOnnxDestroyOfflineSpeakerDiarization(sd);
    return -1;
  }

  const SherpaOnnxOfflineSpeakerDiarizationResult *result =
      SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
          sd, wave.samples.data(), wave.samples.size(), ProgressCallback,
          nullptr);
  if (!result) {
    std::cerr << "Failed to do speaker diarization\n";
    SherpaOnnxDestroyOfflineSpeakerDiarization(sd);
    return -1;
  }

  int32_t num_segments =
      SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);

  const SherpaOnnxOfflineSpeakerDiarizationSegment *segments =
      SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);

  for (int32_t i = 0; i != num_segments; ++i) {
    fprintf(stderr, "%.3f -- %.3f speaker_%02d\n", segments[i].start,
            segments[i].end, segments[i].speaker);
  }

  SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments);
  SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);
  SherpaOnnxDestroyOfflineSpeakerDiarization(sd);

  return 0;
}
