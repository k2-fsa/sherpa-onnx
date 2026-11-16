// c-api-examples/audio8-asr-c-api.c
//
// Copyright (c)  2026  yuyun2000

//
// This file demonstrates how to use Audio8-ASR-0.1B with sherpa-onnx's C API.
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-audio8-asr-0.1b-int8.tar.bz2
tar xvf sherpa-onnx-audio8-asr-0.1b-int8.tar.bz2
rm sherpa-onnx-audio8-asr-0.1b-int8.tar.bz2
cd sherpa-onnx-audio8-asr-0.1b-int8
cp encoder.int8.onnx encoder.onnx
cp decoder.int8.onnx decoder.onnx
cd -
*/
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  // clang-format off
  const char *dir = "./sherpa-onnx-audio8-asr-0.1b-int8";
  char wav_filename[256];
  char encoder_filename[256];
  char decoder_filename[256];
  char tokenizer_dir[256];
  snprintf(wav_filename, sizeof(wav_filename), "%s/test_wavs/0.wav", dir);
  snprintf(encoder_filename, sizeof(encoder_filename), "%s/encoder.onnx", dir);
  snprintf(decoder_filename, sizeof(decoder_filename), "%s/decoder.onnx", dir);
  snprintf(tokenizer_dir, sizeof(tokenizer_dir), "%s/tokenizer", dir);
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
