// cxx-api-examples/pocket-tts-en-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

// This file shows how to use sherpa-onnx CXX API
// for English TTS with PocketTTS.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
tar xf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

./pocket-tts-en-cxx-api

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

  config.model.pocket.lm_flow =
      "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx";
  config.model.pocket.lm_main =
      "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx";
  config.model.pocket.encoder =
      "./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx";
  config.model.pocket.decoder =
      "./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx";
  config.model.pocket.text_conditioner =
      "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx";
  config.model.pocket.vocab_json =
      "./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json";
  config.model.pocket.token_scores_json =
      "./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json";

  config.model.num_threads = 2;

  // If you don't want to see debug messages, please set it to 0
  config.model.debug = 1;

  std::string filename = "./generated-pocket-en-cxx.wav";
  std::string text =
      "Today as always, men fall into two groups: slaves and free men. Whoever "
      "does not have two-thirds of his day for himself, is a slave, whatever "
      "he may be: a statesman, a businessman, an official, or a scholar. "
      "Friends fell out often because life was changing so fast. The easiest "
      "thing in the world was to lose touch with someone.";

  auto tts = OfflineTts::Create(config);
  GenerationConfig cfg;
  cfg.speed = 1.0;

  std::string reference_audio_file =
      "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav";

  Wave wave = ReadWave(reference_audio_file);
  cfg.reference_audio = std::move(wave.samples);
  cfg.reference_sample_rate = wave.sample_rate;
  cfg.extra["max_reference_audio_len"] = "10";

#if 0
  // If you don't want to use a callback, then please enable this branch
  GeneratedAudio audio = tts.Generate(text, cfg);
#else
  GeneratedAudio audio = tts.Generate(text, cfg, ProgressCallback);
#endif

  WriteWave(filename, {audio.samples, audio.sample_rate});

  fprintf(stderr, "Input text is: %s\n", text.c_str());
  fprintf(stderr, "Saved to: %s\n", filename.c_str());

  return 0;
}
