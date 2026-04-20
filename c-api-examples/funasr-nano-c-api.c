// c-api-examples/funasr-nano-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to use FunASR Nano with sherpa-onnx's C API.
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
// tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
// rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  // clang-format off
  const char *wav_filename = "./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/dia_yue.wav";
  const char *encoder_adaptor = "./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx";
  const char *embedding = "./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx";
  const char *llm = "./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx";
  const char *tokenizer = "./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B";
  // clang-format on

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  SherpaOnnxOfflineFunASRNanoModelConfig funasr_nano;
  memset(&funasr_nano, 0, sizeof(funasr_nano));
  funasr_nano.encoder_adaptor = encoder_adaptor;
  funasr_nano.embedding = embedding;
  funasr_nano.llm = llm;
  funasr_nano.tokenizer = tokenizer;

  // Offline model config
  SherpaOnnxOfflineModelConfig offline_model_config;
  memset(&offline_model_config, 0, sizeof(offline_model_config));
  offline_model_config.debug = 1;
  offline_model_config.num_threads = 2;
  offline_model_config.provider = "cpu";
  offline_model_config.funasr_nano = funasr_nano;

  // Recognizer config
  SherpaOnnxOfflineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "greedy_search";
  recognizer_config.model_config = offline_model_config;

  const SherpaOnnxOfflineRecognizer *recognizer =
      SherpaOnnxCreateOfflineRecognizer(&recognizer_config);

  if (recognizer == NULL) {
    fprintf(stderr, "Please check your config!\n");
    SherpaOnnxFreeWave(wave);
    return -1;
  }

  const SherpaOnnxOfflineStream *stream =
      SherpaOnnxCreateOfflineStream(recognizer);

  SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                  wave->num_samples);
  SherpaOnnxDecodeOfflineStream(recognizer, stream);
  const SherpaOnnxOfflineRecognizerResult *result =
      SherpaOnnxGetOfflineStreamResult(stream);

  fprintf(stderr, "Decoded text: %s\n", result->text);

  SherpaOnnxDestroyOfflineRecognizerResult(result);
  SherpaOnnxDestroyOfflineStream(stream);
  SherpaOnnxDestroyOfflineRecognizer(recognizer);
  SherpaOnnxFreeWave(wave);

  return 0;
}
