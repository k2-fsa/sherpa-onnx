// cxx-api-examples/qwen3-asr-cxx-api.cc
//
// Copyright (c)  2026  zengyw
//
// Offline Qwen3-ASR using sherpa-onnx C++ API wrapper.
//
// clang-format off
//
// Build:
//   cmake --build build --target qwen3-asr-cxx-api
//
// Model:
//   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
//   tar xvf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
//
// Run:
//   ./build/bin/qwen3-asr-cxx-api
//
// Hotwords (optional): comma-separated in qwen3.hotwords, e.g. "foo,bar".
//
// Note: If the input audio is too long, you can change max_new_tokens via
//   stream.SetOption("max_new_tokens", "256");
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
  config.model_config.qwen3_asr.conv_frontend = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx";
  config.model_config.qwen3_asr.encoder = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx";
  config.model_config.qwen3_asr.decoder = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx";
  config.model_config.qwen3_asr.tokenizer = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer";
  config.model_config.qwen3_asr.hotwords = "";  // optional; e.g. "foo,bar"
  config.model_config.qwen3_asr.max_total_len = 512;
  config.model_config.qwen3_asr.max_new_tokens = 128;
  config.model_config.qwen3_asr.temperature = 1e-6f;
  config.model_config.qwen3_asr.top_p = 0.8f;
  config.model_config.qwen3_asr.seed = 42;
  // clang-format on

  std::string wave_filename =
      "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav";
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
  // stream.SetOption("language", "Korean");  // optional: force transcription language
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
