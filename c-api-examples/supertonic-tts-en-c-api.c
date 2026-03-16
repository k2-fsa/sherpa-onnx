// c-api-examples/supertonic-tts-en-c-api.c
//
// Copyright (c)  2026  zengyw

// This file shows how to use sherpa-onnx C API
// for English TTS with Supertonic.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
tar xf sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
rm sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2

./supertonic-tts-en-c-api

*/
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

static int32_t ProgressCallback(const float* samples, int32_t num_samples,
                                float progress, void* arg) {
  fprintf(stderr, "Progress: %.3f%%\n", progress * 100);
  // return 1 to continue generating
  // return 0 to stop generating
  return 1;
}

int32_t main(int32_t argc, char* argv[]) {
  SherpaOnnxOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));
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

  const char* filename = "./generated-supertonic-en-c.wav";
  const char* text =
      "Today as always, men fall into two groups: slaves and free men. Whoever "
      "does not have two-thirds of his day for himself, is a slave, whatever "
      "he may be: a statesman, a businessman, an official, or a scholar.";

  const SherpaOnnxOfflineTts* tts = SherpaOnnxCreateOfflineTts(&config);
  if (!tts) {
    fprintf(stderr, "Error create Offline TTS\n");
    return -1;
  }

  SherpaOnnxGenerationConfig cfg = {0};
  cfg.sid = 6;
  cfg.num_steps = 5;
  cfg.speed = 1.25f;  // larger -> faster
  cfg.extra = "{\"lang\": \"en\"}";

  const SherpaOnnxGeneratedAudio* audio =
      SherpaOnnxOfflineTtsGenerateWithConfig(tts, text, &cfg, ProgressCallback,
                                             NULL);

  fprintf(stderr, "Input text is: %s\n", text);

  if (audio) {
    SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, filename);
    fprintf(stderr, "Saved to: %s\n", filename);
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  }

  SherpaOnnxDestroyOfflineTts(tts);

  return 0;
}
