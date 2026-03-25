// c-api-examples/qwen3-asr-c-api.c
//
// Copyright (c)  2026  zengyw
//
// Offline Qwen3-ASR using sherpa-onnx C API (conv_frontend + encoder + decoder
// with KV cache; tokenizer directory).
//
// clang-format off
//
// Build:
//   cmake --build build --target qwen3-asr-c-api
//
// Model:
//   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
//   tar xvf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
//
// Run:
//   ./build/bin/qwen3-asr-c-api
//
// Note: If the input audio is too long, you can set option on the stream:
//   SherpaOnnxOfflineStreamSetOption(stream, "max_new_tokens", "256");
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  // clang-format off
  const char *wav_filename = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav";
  const char *conv_frontend = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx";
  const char *encoder = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx";
  const char *decoder = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx";
  const char *tokenizer = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer";
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
  qwen3.max_new_tokens = 128;
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
