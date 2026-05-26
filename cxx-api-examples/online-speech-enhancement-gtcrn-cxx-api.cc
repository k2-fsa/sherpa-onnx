// cxx-api-examples/online-speech-enhancement-gtcrn-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// We assume you have pre-downloaded the GTCRN model and sample test wave from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
// An example command to download:
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
*/
// clang-format on

#include <chrono>  // NOLINT
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  OnlineSpeechDenoiserConfig config;
  std::string model_filename = "./gtcrn_simple.onnx";
  std::string wav_filename = "./inp_16k.wav";
  std::string out_wave_filename = "./enhanced-online-gtcrn.wav";
  config.model.gtcrn.model = model_filename;

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
