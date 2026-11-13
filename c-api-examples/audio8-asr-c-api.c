// c-api-examples/audio8-asr-c-api.c
//
// Copyright (c)  2026  yuyun2000

//
// This file demonstrates how to use Audio8-ASR-0.1B with sherpa-onnx's C API.
// clang-format off
/*
wget https://github.com/yuyun2000/sherpa-onnx/releases/download/asr-models/sherpa-onnx-audio8-asr-0.1B.tar.bz2
tar xvf sherpa-onnx-audio8-asr-0.1B.tar.bz2
rm sherpa-onnx-audio8-asr-0.1B.tar.bz2
*/
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  // clang-format off
  const char *wav_filename = "./sherpa-onnx-audio8-asr-0.1B/test_wavs/0.wav";
  const char *encoder_filename = "./sherpa-onnx-audio8-asr-0.1B/encoder.onnx";
  const char *decoder_filename = "./sherpa-onnx-audio8-asr-0.1B/decoder.onnx";
  const char *tokenizer_dir = "./sherpa-onnx-audio8-asr-0.1B/tokenizer";
  // clang-format on

  const char *provider = "cpu";

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  SherpaOnnxOfflineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "greedy_search";
  recognizer_config.model_config.debug = 1;
  recognizer_config.model_config.num_threads = 1;
  recognizer_config.model_config.provider = provider;
  recognizer_config.model_config.audio8_asr.encoder = encoder_filename;
  recognizer_config.model_config.audio8_asr.decoder = decoder_filename;
  recognizer_config.model_config.audio8_asr.tokenizer = tokenizer_dir;
  recognizer_config.model_config.audio8_asr.max_total_len = 512;
  recognizer_config.model_config.audio8_asr.max_new_tokens = 128;

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
