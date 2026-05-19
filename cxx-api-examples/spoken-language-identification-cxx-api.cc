// cxx-api-examples/spoken-language-identification-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to use spoken language identification
// with sherpa-onnx's C API (called from C++ code).
//
// We assume you have pre-downloaded the whisper multi-lingual models
// from https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
// An example command to download the "tiny" whisper model is given below:
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
// tar xvf sherpa-onnx-whisper-tiny.tar.bz2
// rm sherpa-onnx-whisper-tiny.tar.bz2
//
// clang-format on

#include <chrono>  // NOLINT
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  SherpaOnnxSpokenLanguageIdentificationConfig config;
  memset(&config, 0, sizeof(config));

  config.whisper.encoder =
      "./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx";
  config.whisper.decoder =
      "./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx";
  config.num_threads = 1;
  config.debug = 1;
  config.provider = "cpu";

  std::cout << "Loading model\n";
  const SherpaOnnxSpokenLanguageIdentification *slid =
      SherpaOnnxCreateSpokenLanguageIdentification(&config);
  if (!slid) {
    std::cerr << "Failed to create spoken language identifier\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  // You can find more test waves from
  // https://hf-mirror.com/spaces/k2-fsa/spoken-language-identification/tree/main/test_wavs
  std::string wav_filename = "./sherpa-onnx-whisper-tiny/test_wavs/0.wav";
  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename.c_str());
  if (wave == nullptr) {
    std::cerr << "Failed to read: '" << wav_filename << "'\n";
    SherpaOnnxDestroySpokenLanguageIdentification(slid);
    return -1;
  }

  SherpaOnnxOfflineStream *stream =
      SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(slid);

  SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                  wave->num_samples);

  std::cout << "Start identification\n";
  const auto begin = std::chrono::steady_clock::now();

  const SherpaOnnxSpokenLanguageIdentificationResult *result =
      SherpaOnnxSpokenLanguageIdentificationCompute(slid, stream);

  const auto end = std::chrono::steady_clock::now();
  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  std::cout << "wav_filename: " << wav_filename << "\n";
  std::cout << "Detected language: " << result->lang << "\n";
  printf("Elapsed seconds: %.3fs\n", elapsed_seconds);

  SherpaOnnxDestroySpokenLanguageIdentificationResult(result);
  SherpaOnnxDestroyOfflineStream(stream);
  SherpaOnnxFreeWave(wave);
  SherpaOnnxDestroySpokenLanguageIdentification(slid);

  return 0;
}
