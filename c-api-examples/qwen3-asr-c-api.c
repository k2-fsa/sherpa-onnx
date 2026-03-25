// c-api-examples/qwen3-asr-c-api.c
//
// Copyright (c)  2026  zengyw
//
// Offline Qwen3-ASR using sherpa-onnx C API (conv_frontend + encoder + decoder
// with KV cache; tokenizer directory).
//
// clang-format off
//
// Prepare a local model directory with:
//   conv_frontend.onnx, encoder.onnx, decoder.onnx, tokenizer/ (vocab.json, ...)
// and a 16-bit PCM mono WAV, then adjust paths below.
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  // clang-format off
  const char *wav_filename = "./test.wav";
  const char *conv_frontend = "./model/conv_frontend.onnx";
  const char *encoder = "./model/encoder.onnx";
  const char *decoder = "./model/decoder.onnx";
  const char *tokenizer = "./model/tokenizer";
  // clang-format on

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  SherpaOnnxOfflineQwen3ASRModelConfig qwen3;
  memset(&qwen3, 0, sizeof(qwen3));
  qwen3.conv_frontend = conv_frontend;
  qwen3.encoder = encoder;
  qwen3.decoder = decoder;
  qwen3.tokenizer = tokenizer;
  qwen3.max_total_len = 512;
  qwen3.max_new_tokens = 64;
  qwen3.temperature = 1e-6f;
  qwen3.top_p = 0.8f;
  qwen3.seed = 42;

  SherpaOnnxOfflineModelConfig offline_model_config;
  memset(&offline_model_config, 0, sizeof(offline_model_config));
  offline_model_config.debug = 1;
  offline_model_config.num_threads = 2;
  offline_model_config.provider = "cpu";
  offline_model_config.qwen3_asr = qwen3;

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
