// cxx-api-examples/moonshine-v2-cxx-api.cc
// Copyright (c)  2024-2026  Xiaomi Corporation

//
// This file demonstrates how to use Moonshine v2 with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
// tar xvf sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
// rm sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
//
// clang-format on

#include <chrono>  // NOLINT
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT
  OfflineRecognizerConfig config;

  // clang-format off
  config.model_config.moonshine.encoder = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort";
  config.model_config.moonshine.merged_decoder = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/decoder_model_merged.ort";
  config.model_config.tokens = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/tokens.txt";
  // clang-format on

  config.model_config.num_threads = 2;

  std::cout << "Loading model\n";
  OfflineRecognizer recognizer = OfflineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename =
      "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/test_wavs/0.wav";
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
