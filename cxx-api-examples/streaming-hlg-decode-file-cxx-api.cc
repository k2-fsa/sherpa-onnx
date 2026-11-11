// cxx-api-examples/streaming-hlg-decode-file-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// This file demonstrates how to decode a streaming CTC model with an HLG
// decoding graph using sherpa-onnx's C++ API.
//
// The HLG graph is passed to the C API through ctc_fst_decoder_config.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
// tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
// rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
//
// clang-format on

#include <chrono>  // NOLINT
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT
  OnlineRecognizerConfig config;

  std::string dir = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18";

  config.model_config.zipformer2_ctc.model =
      dir + "/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx";
  config.model_config.tokens = dir + "/tokens.txt";
  config.model_config.num_threads = 1;
  config.model_config.debug = 0;

  // This is what the example exercises: the graph and max_active reach the C
  // API through ctc_fst_decoder_config.
  config.ctc_fst_decoder_config.graph = dir + "/HLG.fst";
  config.ctc_fst_decoder_config.max_active = 3000;

  std::cout << "Loading model\n";
  OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename = dir + "/test_wavs/8k.wav";
  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  std::cout << "Start recognition\n";
  const auto begin = std::chrono::steady_clock::now();

  OnlineStream stream = recognizer.CreateStream();

  // simulate streaming. You can choose an arbitrary N
  const int32_t N = 3200;
  int32_t segment_id = 0;

  int32_t k = 0;
  while (k < static_cast<int32_t>(wave.samples.size())) {
    int32_t start = k;
    int32_t end = (start + N > static_cast<int32_t>(wave.samples.size()))
                      ? wave.samples.size()
                      : (start + N);
    k += N;

    stream.AcceptWaveform(wave.sample_rate, wave.samples.data() + start,
                          end - start);
    while (recognizer.IsReady(&stream)) {
      recognizer.Decode(&stream);
    }

    OnlineRecognizerResult result = recognizer.GetResult(&stream);
    if (!result.text.empty()) {
      std::cout << "Segment " << segment_id << ": " << result.text << "\n";
    }

    if (recognizer.IsEndpoint(&stream)) {
      if (!result.text.empty()) {
        ++segment_id;
      }
      recognizer.Reset(&stream);
    }
  }

  // add some tail padding
  std::vector<float> tail_paddings(
      static_cast<int32_t>(wave.sample_rate * 0.3), 0.0f);
  stream.AcceptWaveform(wave.sample_rate, tail_paddings.data(),
                        tail_paddings.size());

  stream.InputFinished();
  while (recognizer.IsReady(&stream)) {
    recognizer.Decode(&stream);
  }

  OnlineRecognizerResult result = recognizer.GetResult(&stream);
  if (!result.text.empty()) {
    std::cout << "Segment " << segment_id << ": " << result.text << "\n";
  }

  // The graph's output labels are in the json field, under "words".
  std::cout << "json: " << result.json << "\n";

  const auto end = std::chrono::steady_clock::now();
  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = wave.samples.size() / static_cast<float>(wave.sample_rate);
  float rtf = elapsed_seconds / duration;

  printf("Number of threads: %d\n", config.model_config.num_threads);
  printf("Duration: %.3fs\n", duration);
  printf("Elapsed seconds: %.3fs\n", elapsed_seconds);
  printf("(Real time factor) RTF = %.3f / %.3f = %.3f\n", elapsed_seconds,
         duration, rtf);

  return 0;
}
