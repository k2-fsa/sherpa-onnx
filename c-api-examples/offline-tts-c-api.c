// c-api-examples/offline-tts-c-api.c
//
// Copyright (c)  2023  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// for offline text-to-speech.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-ljs.tar.bz2
tar xf vits-ljs.tar.bz2
rm vits-ljs.tar.bz2

./offline-tts-c-api

 */
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  const char *model = "./vits-ljs/vits-ljs.onnx";
  const char *lexicon = "./vits-ljs/lexicon.txt";
  const char *tokens = "./vits-ljs/tokens.txt";
  const char *filename = "./generated-offline-tts-c.wav";
  const char *text =
      "Today as always, men fall into two groups: slaves and free men. "
      "Whoever does not have two-thirds of his day for himself, is a slave, "
      "whatever he may be: a statesman, a businessman, an official, or a "
      "scholar.";

  SherpaOnnxOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));
  config.model.vits.model = model;
  config.model.vits.lexicon = lexicon;
  config.model.vits.tokens = tokens;
  config.model.num_threads = 1;
  config.model.provider = "cpu";
  config.model.debug = 0;

  const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);
  if (tts == NULL) {
    fprintf(stderr, "Please check your config!\n");
    return -1;
  }

  SherpaOnnxGenerationConfig cfg = {0};
  cfg.silence_scale = 0.2f;
  cfg.sid = 0;
  cfg.speed = 1.0f;

  const SherpaOnnxGeneratedAudio *audio =
      SherpaOnnxOfflineTtsGenerateWithConfig(tts, text, &cfg, NULL, NULL);
  if (audio == NULL) {
    fprintf(stderr, "Failed to generate audio!\n");
    SherpaOnnxDestroyOfflineTts(tts);
    return -1;
  }

  SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, filename);

  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  SherpaOnnxDestroyOfflineTts(tts);

  fprintf(stderr, "Input text is: %s\n", text);
  fprintf(stderr, "Speaker ID is: %d\n", cfg.sid);
  fprintf(stderr, "Saved to: %s\n", filename);

  return 0;
}
