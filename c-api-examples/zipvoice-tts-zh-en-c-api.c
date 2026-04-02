// c-api-examples/zipvoice-tts-zh-en-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// for Chinese/English zero-shot TTS with ZipVoice.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
rm sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

./zipvoice-tts-zh-en-c-api
*/
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

static int32_t ProgressCallback(const float *samples, int32_t num_samples,
                                float progress, void *arg) {
  fprintf(stderr, "Progress: %.3f%%\n", progress * 100);
  return 1;
}

int32_t main(int32_t argc, char *argv[]) {
  SherpaOnnxOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));
  config.model.zipvoice.encoder =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx";
  config.model.zipvoice.decoder =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx";
  config.model.zipvoice.data_dir =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data";
  config.model.zipvoice.lexicon =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt";
  config.model.zipvoice.tokens =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt";
  config.model.zipvoice.vocoder = "./vocos_24khz.onnx";

  config.model.num_threads = 2;

  // If you want to see more debug messages, please set it to 1
  config.model.debug = 0;

  const char *filename = "./generated-zipvoice-zh-en-c.wav";
  const char *text =
      "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, "
      "就是全心投入并享受其中.";
  const char *reference_text =
      "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系.";
  const char *reference_audio_file =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav";

  const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);
  if (!tts) {
    fprintf(stderr, "Error create Offline TTS\n");
    return -1;
  }

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(reference_audio_file);
  if (!wave) {
    fprintf(stderr, "Failed to read %s\n", reference_audio_file);
    SherpaOnnxDestroyOfflineTts(tts);
    return -1;
  }

  SherpaOnnxGenerationConfig cfg = {0};
  cfg.speed = 1.0f;
  cfg.num_steps = 4;
  cfg.reference_audio = wave->samples;
  cfg.reference_audio_len = wave->num_samples;
  cfg.reference_sample_rate = wave->sample_rate;
  cfg.reference_text = reference_text;
  cfg.extra = "{\"min_char_in_sentence\": 10}";

#if 0
  // If you don't want to use a callback, then please enable this branch
  const SherpaOnnxGeneratedAudio *audio =
      SherpaOnnxOfflineTtsGenerateWithConfig(tts, text, &cfg, NULL, NULL);
#else
  const SherpaOnnxGeneratedAudio *audio =
      SherpaOnnxOfflineTtsGenerateWithConfig(tts, text, &cfg, ProgressCallback,
                                             NULL);
#endif

  SherpaOnnxFreeWave(wave);

  fprintf(stderr, "Input text is: %s\n", text);

  if (audio) {
    SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, filename);
    fprintf(stderr, "Saved to: %s\n", filename);
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  }

  SherpaOnnxDestroyOfflineTts(tts);

  return 0;
}
