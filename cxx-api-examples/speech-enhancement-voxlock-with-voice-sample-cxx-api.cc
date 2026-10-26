// cxx-api-examples/speech-enhancement-voxlock-with-voice-sample-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of
// one speaker, e.g., the caller in a phone call, and removes the other
// voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example provides a voice sample of the speaker to keep, so VoxLock
// isolates that speaker from the first sample. The voice sample must have at
// least 3 seconds of the speaker's voice; from a longer file, VoxLock uses
// the loudest 3 seconds within its first 30 seconds. See
// ./speech-enhancement-voxlock-cxx-api.cc if you don't have a voice sample.
//
// We assume you have pre-downloaded the model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
// An example command to download:
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
tar xvf sherpa-onnx-voxlock-int8.tar.bz2
*/
// clang-format on

#include <chrono>  // NOLINT
#include <cstdio>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  OfflineSpeechDenoiserConfig config;
  std::string enroll_model = "./sherpa-onnx-voxlock-int8/enroll.onnx";
  std::string chunk_model = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx";
  std::string voice_sample =
      "./sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav";
  std::string wav_filename =
      "./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav";
  std::string out_wave_filename =
      "./voxlock-enhanced-with-voice-sample-16k.wav";
  config.model.voxlock.enroll_model = enroll_model;
  config.model.voxlock.chunk_model = chunk_model;
  config.model.voxlock.voice_sample = voice_sample;

  auto sd = OfflineSpeechDenoiser::Create(config);
  if (!sd.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }

  Wave wave = ReadWave(wav_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wav_filename << "'\n";
    return -1;
  }

  std::cout << "Started\n";
  const auto begin = std::chrono::steady_clock::now();
  auto denoised =
      sd.Run(wave.samples.data(), wave.samples.size(), wave.sample_rate);
  const auto end = std::chrono::steady_clock::now();
  std::cout << "Done\n";

  WriteWave(out_wave_filename, {denoised.samples, denoised.sample_rate});

  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = wave.samples.size() / static_cast<float>(wave.sample_rate);
  float rtf = elapsed_seconds / duration;

  std::cout << "Saved to " << out_wave_filename << "\n";
  printf("Duration: %.3fs\n", duration);
  printf("Elapsed seconds: %.3fs\n", elapsed_seconds);
  printf("(Real time factor) RTF = %.3f / %.3f = %.3f\n", elapsed_seconds,
         duration, rtf);
  return 0;
}
