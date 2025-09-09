// cxx-api-examples/streaming-t-one-ctc-cxx-api.ccc
// Copyright (c)  2025  Xiaomi Corporation

//
// This file demonstrates how to use streaming T-one
// with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
// tar xvf sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
// rm sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
//
// clang-format on

#include <chrono>  // NOLINT
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT
  OnlineRecognizerConfig config;

  // please see
  config.model_config.t_one_ctc.model =
      "sherpa-onnx-streaming-t-one-russian-2025-09-08/model.onnx";

  config.model_config.tokens =
      "sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt";

  config.model_config.num_threads = 1;

  std::cout << "Loading model\n";
  OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename =
      "sherpa-onnx-streaming-t-one-russian-2025-09-08/0.wav";

  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  std::cout << "Start recognition\n";
  const auto begin = std::chrono::steady_clock::now();

  OnlineStream stream = recognizer.CreateStream();
  std::vector<float> left_padding(2400);  // 0.3 seconds at 8kHz
  std::vector<float> tail_padding(4800);  // 0.6 seconds at 8kHz

  stream.AcceptWaveform(wave.sample_rate, left_padding.data(),
                        left_padding.size());
  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());
  stream.AcceptWaveform(wave.sample_rate, tail_padding.data(),
                        tail_padding.size());
  stream.InputFinished();

  while (recognizer.IsReady(&stream)) {
    recognizer.Decode(&stream);
  }

  OnlineRecognizerResult result = recognizer.GetResult(&stream);

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
