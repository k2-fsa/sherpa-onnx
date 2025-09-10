// cxx-api-examples/piper-tts-cxx-api.cc
//
// Copyright (c)  2025  Xiaomi Corporation

// This file shows how to use sherpa-onnx CXX API
// for TTS with Piper models.
//
// clang-format off
/*
Usage

wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-amy-low.tar.gz
tar xf voice-en-us-amy-low.tar.gz
rm voice-en-us-amy-low.tar.gz

./piper-tts-cxx-api

 */
// clang-format on

#include <cstdint>
#include <cstdio>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

static int32_t ProgressCallback(const float *samples, int32_t num_samples,
                                float progress, void *arg) {
  fprintf(stderr, "Progress: %.3f%%\n", progress * 100);
  // return 1 to continue generating
  // return 0 to stop generating
  return 1;
}

int32_t main(int32_t argc, char *argv[]) {
  using namespace sherpa_onnx::cxx;  // NOLINT
  OfflineTtsConfig config;

  config.model.piper.model = "./en_US-amy-low.onnx";
  config.model.piper.model_config_file = "./en_US-amy-low.onnx.json";
  config.model.piper.data_dir = "./espeak-ng-data";

  config.model.num_threads = 2;

  // If you don't want to see debug messages, please set it to 0
  config.model.debug = 1;

  std::string filename = "./generated-piper-cxx.wav";
  std::string text = 
      "Welcome to Piper text-to-speech synthesis. "
      "This is a high-quality neural voice that can speak multiple languages. "
      "The output sounds natural and expressive.";

  auto tts = OfflineTts::Create(config);
  int32_t sid = 0;
  float speed = 1.0;  // larger -> faster in speech speed

#if 0
  // If you don't want to use a callback, then please enable this branch
  GeneratedAudio audio = tts.Generate(text, sid, speed);
#else
  GeneratedAudio audio = tts.Generate(text, sid, speed, ProgressCallback);
#endif

  WriteWave(filename, {audio.samples, audio.sample_rate});

  fprintf(stderr, "Input text is: %s\n", text.c_str());
  fprintf(stderr, "Speaker ID is: %d\n", sid);
  fprintf(stderr, "Saved to: %s\n", filename.c_str());

  return 0;
}