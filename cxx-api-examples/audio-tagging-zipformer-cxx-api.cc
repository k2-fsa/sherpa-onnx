// cxx-api-examples/audio-tagging-zipformer-cxx-api.cc
// Copyright (c)  2025  Xiaomi Corporation

//
// This file demonstrates how to use Zipformer with sherpa-onnx's C++
// API for audio tagging.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
// tar xvf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
// rm sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
//
//
// clang-format on
#include <chrono>  // NOLINT
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT
  AudioTaggingConfig config;

  config.model.zipformer.model =
      "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.onnx";
  config.model.num_threads = 1;
  config.model.debug = true;
  config.labels =
      "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/"
      "class_labels_indices.csv";

  config.top_k = 5;

  std::cout << "Loading model\n";
  AudioTagging tagger = AudioTagging::Create(config);
  if (!tagger.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }

  std::string wave_filename =
      "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/1.wav";

  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  std::cout << "Started\n";
  const auto begin = std::chrono::steady_clock::now();

  OfflineStream stream = tagger.CreateStream();
  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());
  std::vector<AudioEvent> events = tagger.Compute(&stream);

  const auto end = std::chrono::steady_clock::now();
  std::cout << "Done\n";

  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = wave.samples.size() / static_cast<float>(wave.sample_rate);
  float rtf = elapsed_seconds / duration;

  int32_t i = 0;

  for (const auto &event : events) {
    fprintf(stderr, "%d: AudioEvent(name='%s', index=%d, prob=%.3f)\n", i,
            event.name.c_str(), event.index, event.prob);
    i += 1;
  }

  printf("Number of threads: %d\n", config.model.num_threads);
  printf("Duration: %.3fs\n", duration);
  printf("Elapsed seconds: %.3fs\n", elapsed_seconds);
  printf("(Real time factor) RTF = %.3f / %.3f = %.3f\n", elapsed_seconds,
         duration, rtf);
}
