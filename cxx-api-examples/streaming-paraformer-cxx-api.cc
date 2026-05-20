// cxx-api-examples/streaming-paraformer-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to use streaming Paraformer
// with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
// tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
// rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
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

  config.model_config.paraformer.encoder =
      "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx";
  config.model_config.paraformer.decoder =
      "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx";
  config.model_config.tokens =
      "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt";

  config.model_config.num_threads = 1;
  config.model_config.debug = true;
  config.enable_endpoint = true;

  std::cout << "Loading model\n";
  OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename =
      "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/0.wav";
  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  std::cout << "Start recognition\n";
  const auto begin = std::chrono::steady_clock::now();

  OnlineStream stream = recognizer.CreateStream();

  // simulate streaming. You can choose an arbitrary N
  const int32_t n = 3200;
  int32_t k = 0;
  int32_t segment_id = 0;

  fprintf(stderr, "sample rate: %d, num samples: %d, duration: %.2f s\n",
          wave.sample_rate, static_cast<int32_t>(wave.samples.size()),
          static_cast<float>(wave.samples.size()) / wave.sample_rate);

  while (k < static_cast<int32_t>(wave.samples.size())) {
    int32_t start = k;
    int32_t end = std::min(start + n, static_cast<int32_t>(wave.samples.size()));
    k += n;

    stream.AcceptWaveform(wave.sample_rate, wave.samples.data() + start,
                          end - start);
    while (recognizer.IsReady(&stream)) {
      recognizer.Decode(&stream);
    }

    OnlineRecognizerResult result = recognizer.GetResult(&stream);

    if (!result.text.empty()) {
      fprintf(stderr, "%d: %s\n", segment_id, result.text.c_str());
    }

    if (recognizer.IsEndpoint(&stream)) {
      if (!result.text.empty()) {
        ++segment_id;
      }
      recognizer.Reset(&stream);
    }
  }

  // add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  stream.AcceptWaveform(wave.sample_rate, tail_paddings, 4800);

  stream.InputFinished();
  while (recognizer.IsReady(&stream)) {
    recognizer.Decode(&stream);
  }

  OnlineRecognizerResult result = recognizer.GetResult(&stream);
  if (!result.text.empty()) {
    fprintf(stderr, "%d: %s\n", segment_id, result.text.c_str());
  }

  const auto end_time = std::chrono::steady_clock::now();
  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin)
          .count() /
      1000.;
  float duration = wave.samples.size() / static_cast<float>(wave.sample_rate);
  float rtf = elapsed_seconds / duration;

  printf("Number of threads: %d\n", config.model_config.num_threads);
  printf("Duration: %.3fs\n", duration);
  printf("Elapsed seconds: %.3fs\n", elapsed_seconds);
  printf("(Real time factor) RTF = %.3f / %.3f = %.3f\n", elapsed_seconds,
         duration, rtf);

  fprintf(stderr, "\n");

  return 0;
}
