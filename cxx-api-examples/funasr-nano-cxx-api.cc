// cxx-api-examples/funasr-nano-cxx-api.cc
//
// Copyright (c)  2025  zengyw
//
// This file demonstrates how to use FunASR-nano with sherpa-onnx's C++ API.
//
//
// clang-format off
//
// Usage:
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
// tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
// rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
//
// clang-format on

#include <chrono>
#include <cstring>
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
  config.model_config.funasr_nano.encoder_adaptor = "./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx";
  config.model_config.funasr_nano.llm = "./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx";
  config.model_config.funasr_nano.embedding = "./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx";
  config.model_config.funasr_nano.tokenizer = "./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B";

  // clang-format on

  std::cout << "Loading model\n";
  OfflineRecognizer recognizer = OfflineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename =
      "./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_yue.wav";

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
