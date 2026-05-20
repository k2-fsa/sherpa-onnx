// cxx-api-examples/speaker-identification-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to use speaker embedding extraction and
// speaker identification with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
//
// Also download test data from https://github.com/csukuangfj/sr-data
//
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"

static std::vector<float> ComputeEmbedding(
    const sherpa_onnx::cxx::SpeakerEmbeddingExtractor &ex,
    const std::string &wav_filename) {
  using namespace sherpa_onnx::cxx;  // NOLINT
  Wave wave = ReadWave(wav_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read " << wav_filename << "\n";
    exit(EXIT_FAILURE);
  }

  OnlineStream stream = ex.CreateStream();
  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());
  stream.InputFinished();

  if (!ex.IsReady(&stream)) {
    std::cerr << "The input wave file " << wav_filename << " is too short!\n";
    exit(EXIT_FAILURE);
  }

  return ex.ComputeEmbedding(&stream);
}

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT
  SpeakerEmbeddingExtractorConfig config;
  config.model = "./3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx";
  config.num_threads = 1;

  SpeakerEmbeddingExtractor ex = SpeakerEmbeddingExtractor::Create(config);
  if (!ex.Get()) {
    std::cerr << "Failed to create speaker embedding extractor\n";
    return -1;
  }

  int32_t dim = ex.Dim();
  std::cout << "Embedding dimension: " << dim << "\n";

  SpeakerEmbeddingManager manager = SpeakerEmbeddingManager::Create(dim);

  // Enroll speakers
  std::vector<std::vector<float>> spk1_embeddings;
  spk1_embeddings.push_back(ComputeEmbedding(ex, "./sr-data/enroll/fangjun-sr-1.wav"));
  spk1_embeddings.push_back(ComputeEmbedding(ex, "./sr-data/enroll/fangjun-sr-2.wav"));
  spk1_embeddings.push_back(ComputeEmbedding(ex, "./sr-data/enroll/fangjun-sr-3.wav"));

  std::vector<std::vector<float>> spk2_embeddings;
  spk2_embeddings.push_back(ComputeEmbedding(ex, "./sr-data/enroll/leijun-sr-1.wav"));
  spk2_embeddings.push_back(ComputeEmbedding(ex, "./sr-data/enroll/leijun-sr-2.wav"));

  const float *spk1_vec[4] = {spk1_embeddings[0].data(),
                               spk1_embeddings[1].data(),
                               spk1_embeddings[2].data(), nullptr};
  const float *spk2_vec[3] = {spk2_embeddings[0].data(),
                               spk2_embeddings[1].data(), nullptr};

  manager.AddList("fangjun", spk1_vec);
  manager.AddList("leijun", spk2_vec);

  std::cout << "Enrolled speakers: " << manager.NumSpeakers() << "\n";

  // Search
  float threshold = 0.6f;

  auto v1 = ComputeEmbedding(ex, "./sr-data/test/fangjun-test-sr-1.wav");
  std::string name1 = manager.Search(v1.data(), threshold);
  std::cout << "fangjun-test-sr-1.wav: " << (name1.empty() ? "unknown" : name1)
            << "\n";

  auto v2 = ComputeEmbedding(ex, "./sr-data/test/leijun-test-sr-1.wav");
  std::string name2 = manager.Search(v2.data(), threshold);
  std::cout << "leijun-test-sr-1.wav: " << (name2.empty() ? "unknown" : name2)
            << "\n";

  // Verify
  bool ok = manager.Verify("fangjun", v1.data(), threshold);
  std::cout << "fangjun-test-sr-1.wav matches fangjun: "
            << (ok ? "yes" : "no") << "\n";

  return 0;
}
