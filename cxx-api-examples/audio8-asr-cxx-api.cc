// cxx-api-examples/audio8-asr-cxx-api.cc
// Copyright (c)  2026  yuyun2000

//
// This file demonstrates how to use Audio8-ASR-0.1B with sherpa-onnx's C++ API.
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-audio8-asr-0.1b-int8.tar.bz2
tar xvf sherpa-onnx-audio8-asr-0.1b-int8.tar.bz2
rm sherpa-onnx-audio8-asr-0.1b-int8.tar.bz2
cp sherpa-onnx-audio8-asr-0.1b-int8/encoder.int8.onnx sherpa-onnx-audio8-asr-0.1b-int8/encoder.onnx
cp sherpa-onnx-audio8-asr-0.1b-int8/decoder.int8.onnx sherpa-onnx-audio8-asr-0.1b-int8/decoder.onnx
*/
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

  // clang-format off
  const std::string dir = "./sherpa-onnx-audio8-asr-0.1b-int8";
  config.model_config.audio8_asr.encoder = dir + "/encoder.onnx";
  config.model_config.audio8_asr.decoder = dir + "/decoder.onnx";
  config.model_config.audio8_asr.tokenizer = dir + "/tokenizer";
  config.model_config.audio8_asr.max_new_tokens = 128;

  config.model_config.num_threads = 2;

  std::cout << "Loading model\n";
  OfflineRecognizer recognizer = OfflineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename = dir + "/test_wavs/0.wav";
  // clang-format on

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

  std::cout << "Text: " << result.text << "\n";
  std::cout << "Duration: " << duration << " s\n";
  std::cout << "Elapsed: " << elapsed_seconds << " s\n";
  std::cout << "RTF: " << rtf << "\n";

  return 0;
}
