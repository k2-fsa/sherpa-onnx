// cxx-api-examples/source-separation-uvr-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file demonstrates how to use the source-separation C++ API
// with the UVR (MDX-Net) model.
//
// Usage:
//
// 1. Download the test model and audio
//
//  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx
//  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
//
// 2. Build
//
//  g++ -std=c++17 -o source-separation-uvr-cxx-api \
//    ./cxx-api-examples/source-separation-uvr-cxx-api.cc \
//    -I ./build/install/include \
//    -L ./build/install/lib/ \
//    -l sherpa-onnx-cxx-api -l sherpa-onnx-c-api -l onnxruntime
//
// 3. Run
//
//  ./source-separation-uvr-cxx-api

#include <chrono>  // NOLINT
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  OfflineSourceSeparationConfig config;
  config.model.uvr.model = "./UVR-MDX-NET-Voc_FT.onnx";
  config.model.num_threads = 1;

  auto ss = OfflineSourceSeparation::Create(config);
  if (!ss.Get()) {
    std::cerr << "Failed to create source separation engine\n";
    return -1;
  }

  const auto *wave = SherpaOnnxReadWaveMultiChannel("./qi-feng-le-zh.wav");
  if (!wave) {
    std::cerr << "Failed to read ./qi-feng-le-zh.wav\n";
    return -1;
  }

  std::cout << "Input wave: channels=" << wave->num_channels
            << ", samples_per_channel=" << wave->num_samples
            << ", sample_rate=" << wave->sample_rate << "\n";

  std::cout << "Started\n";
  const auto begin = std::chrono::steady_clock::now();
  auto output = ss.Process(wave->samples, wave->num_channels,
                           wave->num_samples, wave->sample_rate);
  const auto end = std::chrono::steady_clock::now();
  std::cout << "Done\n";

  std::cout << "Output: " << output.stems.size() << " stems, sample_rate="
            << output.sample_rate << "\n";

  // Write each stem to a separate multi-channel wave file
  std::vector<std::string> stem_names = {"uvr-vocals", "uvr-non-vocals"};
  for (int32_t s = 0;
       s < static_cast<int32_t>(output.stems.size()) &&
       s < static_cast<int32_t>(stem_names.size());
       ++s) {
    auto &stem = output.stems[s];
    int32_t nc = static_cast<int32_t>(stem.samples.size());
    int32_t ns = nc > 0 ? static_cast<int32_t>(stem.samples[0].size()) : 0;

    std::vector<const float *> stem_ptrs(nc);
    for (int32_t c = 0; c < nc; ++c) {
      stem_ptrs[c] = stem.samples[c].data();
    }

    std::string filename = stem_names[s] + ".wav";
    SherpaOnnxWriteWaveMultiChannel(stem_ptrs.data(), ns, output.sample_rate,
                                    nc, filename.c_str());
    std::cout << "Saved " << filename << " (" << nc << " channels, " << ns
              << " samples, " << output.sample_rate << " Hz)\n";
  }

  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration =
      wave->num_samples / static_cast<float>(wave->sample_rate);
  float rtf = elapsed_seconds / duration;

  printf("Duration: %.3fs\n", duration);
  printf("Elapsed seconds: %.3fs\n", elapsed_seconds);
  printf("(Real time factor) RTF = %.3f / %.3f = %.3f\n", elapsed_seconds,
         duration, rtf);

  SherpaOnnxFreeMultiChannelWave(wave);
  return 0;
}
