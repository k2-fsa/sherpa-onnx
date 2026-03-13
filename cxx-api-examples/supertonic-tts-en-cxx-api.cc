// cxx-api-examples/supertonic-tts-en-cxx-api.cc
//
// Copyright (c)  2026  zengyw

// This file shows how to use sherpa-onnx CXX API
// for English TTS with Supertonic.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
tar xf sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
rm sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2

./supertonic-tts-en-cxx-api

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

  config.model.supertonic.duration_predictor =
      "./sherpa-onnx-supertonic-tts-int8-2026-03-06/"
      "duration_predictor.int8.onnx";
  config.model.supertonic.text_encoder =
      "./sherpa-onnx-supertonic-tts-int8-2026-03-06/text_encoder.int8.onnx";
  config.model.supertonic.vector_estimator =
      "./sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx";
  config.model.supertonic.vocoder =
      "./sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx";
  config.model.supertonic.tts_json =
      "./sherpa-onnx-supertonic-tts-int8-2026-03-06/tts.json";
  config.model.supertonic.unicode_indexer =
      "./sherpa-onnx-supertonic-tts-int8-2026-03-06/unicode_indexer.bin";
  config.model.supertonic.voice_style =
      "./sherpa-onnx-supertonic-tts-int8-2026-03-06/voice.bin";

  config.model.num_threads = 2;

  // If you don't want to see debug messages, please set it to 0
  config.model.debug = 1;

  std::string filename = "./generated-supertonic-en-cxx.wav";
  std::string text =
      "Today as always, men fall into two groups: slaves and free men. Whoever "
      "does not have two-thirds of his day for himself, is a slave, whatever "
      "he may be: a statesman, a businessman, an official, or a scholar.";

  auto tts = OfflineTts::Create(config);

  GenerationConfig gen_config;
  gen_config.sid = 6;
  gen_config.num_steps = 5;
  gen_config.speed = 1.25;  // larger -> faster
  gen_config.extra["lang"] = "en";

  // Use GenerationConfig for Supertonic.
  GeneratedAudio audio = tts.Generate(text, gen_config, ProgressCallback);

  WriteWave(filename, {audio.samples, audio.sample_rate});

  fprintf(stderr, "Input text is: %s\n", text.c_str());
  fprintf(stderr, "Saved to: %s\n", filename.c_str());

  return 0;
}
