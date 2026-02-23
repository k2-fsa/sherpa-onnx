// c-api-examples/pocket-tts-en-c-api.c
//
// Copyright (c)  2026  Xiaoyingtao Corporation

// This file shows how to use sherpa-onnx C API
// for English TTS with Pocket TTS.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
tar xf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

./pocket-tts-en-c-api

 */
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

static int32_t ProgressCallback(const float *samples, int32_t num_samples,
                                float progress, void *arg) {
  fprintf(stderr, "Progress: %.3f%%\n", progress * 100);
  // return 1 to continue generating
  // return 0 to stop generating
  return 1;
}

int32_t main(int32_t argc, char *argv[]) {
  SherpaOnnxOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));
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
  // Voice embedding cache capacity (default: 50)
  // Increase this if you have many different reference audios to avoid
  // recomputing voice embeddings
  config.model.pocket.voice_embedding_cache_capacity = 50;

  config.model.num_threads = 2;

  // If you don't want to see debug messages, please set it to 0
  config.model.debug = 1;

  const char *filename = "./generated-pocket-en.wav";
  const char *text =
      "Today as always, men fall into two groups: slaves and free men. Whoever "
      "does not have two-thirds of his day for himself, is a slave, whatever "
      "he may be: a statesman, a businessman, an official, or a scholar. "
      "Friends fell out often because life was changing so fast. The easiest "
      "thing in the world was to lose touch with someone.";

  const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);
  if (!tts) {
    fprintf(stderr, "Error create Offline TTS\n");
    return -1;
  }
  float speed = 1.0;  // larger -> faster in speech speed
  SherpaOnnxGenerationConfig cfg = {0};
  const char *reference_audio_file =
      "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav";
  const SherpaOnnxWave *wave = NULL;
  wave = SherpaOnnxReadWave(reference_audio_file);
  if (!wave) {
    fprintf(stderr, "Failed to read %s\n", reference_audio_file);
    SherpaOnnxDestroyOfflineTts(tts);
    return -1;
  }
  cfg.reference_audio = wave->samples;
  cfg.reference_audio_len = wave->num_samples;
  cfg.reference_sample_rate = wave->sample_rate;
  // Extra parameters passed as JSON string
  // - max_reference_audio_len: maximum length of reference audio in seconds
  // - seed: random seed for reproducibility (optional, -1 for random)
  cfg.extra = "{\"max_reference_audio_len\": 10.0, \"seed\": 42}";

#if 0
  // If you don't want to use a callback, then please enable this branch
  const SherpaOnnxGeneratedAudio *audio =
      SherpaOnnxOfflineTtsGenerateWithConfig(tts, text, &cfg, NULL, NULL);
#else
  const SherpaOnnxGeneratedAudio *audio =
      SherpaOnnxOfflineTtsGenerateWithConfig(tts, text, &cfg, ProgressCallback,
                                             NULL);
#endif

  if (wave) SherpaOnnxFreeWave(wave);

  fprintf(stderr, "Input text is: %s\n", text);

  if (audio) {
    SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, filename);
    fprintf(stderr, "Saved to: %s\n", filename);
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  }

  SherpaOnnxDestroyOfflineTts(tts);

  return 0;
}
