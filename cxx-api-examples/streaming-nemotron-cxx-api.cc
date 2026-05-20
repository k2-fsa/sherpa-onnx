// cxx-api-examples/streaming-nemotron-cxx-api.cc
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to use streaming Nemotron
// with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2
// tar xvf sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2
// rm sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2
//
// clang-format on

#include <chrono>  // NOLINT
#include <cstdio>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT
  OnlineRecognizerConfig config;

  config.model_config.transducer.encoder =
      "./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/"
      "encoder.int8.onnx";

  config.model_config.transducer.decoder =
      "./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/"
      "decoder.int8.onnx";

  config.model_config.transducer.joiner =
      "./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/"
      "joiner.int8.onnx";

  config.model_config.tokens =
      "./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/"
      "tokens.txt";

  config.model_config.num_threads = 1;

  std::cout << "Loading model\n";
  OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename =
      "./sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25/"
      "test_wavs/0.wav";
  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  std::cout << "Start recognition\n";
  const auto begin = std::chrono::steady_clock::now();

  OnlineStream stream = recognizer.CreateStream();
  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());
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
