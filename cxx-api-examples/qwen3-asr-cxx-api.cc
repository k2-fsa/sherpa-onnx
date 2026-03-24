// cxx-api-examples/qwen3-asr-cxx-api.cc
//
// Copyright (c)  2026  zengyw
//
// Offline Qwen3-ASR using sherpa-onnx C++ API wrapper.
//
// clang-format off
//
// Usage:
//   Adjust paths to conv_frontend.onnx, encoder.onnx, decoder.onnx, tokenizer/
//   and input WAV, then build and run this example.
//
// clang-format on

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main(int32_t argc, char *argv[]) {
  using namespace sherpa_onnx::cxx;

  OfflineRecognizerConfig config;
  config.model_config.num_threads = 2;
  config.model_config.debug = false;
  config.model_config.provider = "cpu";

  // clang-format off
  config.model_config.qwen3_asr.conv_frontend = "./model/conv_frontend.onnx";
  config.model_config.qwen3_asr.encoder = "./model/encoder.onnx";
  config.model_config.qwen3_asr.decoder = "./model/decoder.onnx";
  config.model_config.qwen3_asr.tokenizer = "./model/tokenizer";
  config.model_config.qwen3_asr.max_total_len = 512;
  config.model_config.qwen3_asr.max_new_tokens = 64;
  config.model_config.qwen3_asr.temperature = 1e-6f;
  config.model_config.qwen3_asr.top_p = 0.8f;
  config.model_config.qwen3_asr.seed = 42;
  // clang-format on

  std::string wave_filename = "./test.wav";
  if (argc >= 2) {
    wave_filename = argv[1];
  }

  std::cout << "Loading model\n";
  OfflineRecognizer recognizer = OfflineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  std::cout << "Start recognition\n";
  const auto begin = std::chrono::steady_clock::now();

  OfflineStream stream = recognizer.CreateStream();
  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());

  recognizer.Decode(&stream);

  OfflineRecognizerResult result = recognizer.GetResult(&stream);

  const auto end = std::chrono::steady_clock::now();
  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = wave.samples.size() / static_cast<float>(wave.sample_rate);
  float rtf = elapsed_seconds / duration;

  std::cout << "text: " << result.text << "\n";
  printf("Number of threads: %d\n", config.model_config.num_threads);
  printf("Duration: %.3fs\n", duration);
  printf("Elapsed seconds: %.3fs\n", elapsed_seconds);
  printf("(Real time factor) RTF = %.3f / %.3f = %.3f\n", elapsed_seconds,
         duration, rtf);

  return 0;
}
