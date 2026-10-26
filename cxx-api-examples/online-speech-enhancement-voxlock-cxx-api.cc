// cxx-api-examples/online-speech-enhancement-voxlock-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of
// one speaker, e.g., the caller in a phone call, and removes the other
// voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock passes the audio
// through unchanged until it has heard 3 seconds of speech, takes that speech
// as the speaker to keep, and isolates from then on. See
// ./online-speech-enhancement-voxlock-with-voice-sample-cxx-api.cc if you
// have a voice sample of the speaker.
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

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  OnlineSpeechDenoiserConfig config;
  std::string enroll_model = "./sherpa-onnx-voxlock-int8/enroll.onnx";
  std::string chunk_model = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx";
  std::string wav_filename =
      "./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav";
  std::string out_wave_filename = "./voxlock-enhanced-online-16k.wav";
  config.model.voxlock.enroll_model = enroll_model;
  config.model.voxlock.chunk_model = chunk_model;

  auto sd = OnlineSpeechDenoiser::Create(config);
  if (!sd.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }

  Wave wave = ReadWave(wav_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wav_filename << "'\n";
    return -1;
  }

  std::vector<float> samples;
  auto frame_shift = sd.GetFrameShiftInSamples();

  std::cout << "Started\n";
  const auto begin = std::chrono::steady_clock::now();

  for (int32_t start = 0; start < static_cast<int32_t>(wave.samples.size());
       start += frame_shift) {
    int32_t n = std::min<int32_t>(frame_shift, wave.samples.size() - start);
    auto denoised = sd.Run(wave.samples.data() + start, n, wave.sample_rate);
    samples.insert(samples.end(), denoised.samples.begin(),
                   denoised.samples.end());
  }

  auto tail = sd.Flush();
  samples.insert(samples.end(), tail.samples.begin(), tail.samples.end());

  const auto end = std::chrono::steady_clock::now();
  std::cout << "Done\n";

  WriteWave(out_wave_filename, {samples, sd.GetSampleRate()});

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
