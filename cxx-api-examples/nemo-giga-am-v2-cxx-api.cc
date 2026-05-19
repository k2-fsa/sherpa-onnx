// cxx-api-examples/nemo-giga-am-v2-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to use the NeMo transducer GigaAM v2 model
// with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2
// tar xvf sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2
// rm sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2
//
// clang-format on

#include <chrono>  // NOLINT
#include <cstdio>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT
  OfflineRecognizerConfig config;

  config.model_config.transducer.encoder =
      "./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/"
      "encoder.int8.onnx";

  config.model_config.transducer.decoder =
      "./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/"
      "decoder.onnx";

  config.model_config.transducer.joiner =
      "./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/"
      "joiner.onnx";

  config.model_config.tokens =
      "./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/"
      "tokens.txt";

  config.model_config.num_threads = 1;

  std::cout << "Loading model\n";
  OfflineRecognizer recognizer = OfflineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename =
      "./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/"
      "test_wavs/example.wav";

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
