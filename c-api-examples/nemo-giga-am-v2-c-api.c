// c-api-examples/nemo-giga-am-v2-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to use the NeMo transducer GigaAM v2 model
// with sherpa-onnx's C API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2
// tar xvf sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2
// rm sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  const char *wav_filename =
      "./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/"
      "test_wavs/example.wav";
  const char *encoder_filename =
      "./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/"
      "encoder.int8.onnx";
  const char *decoder_filename =
      "./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/"
      "decoder.onnx";
  const char *joiner_filename =
      "./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/"
      "joiner.onnx";
  const char *tokens_filename =
      "./sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19/"
      "tokens.txt";
  const char *provider = "cpu";

  if (!SherpaOnnxFileExists(wav_filename)) {
    fprintf(stderr, "File not found: %s\n", wav_filename);
    return -1;
  }
  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read or parse %s\n", wav_filename);
    return -1;
  }

  SherpaOnnxOfflineModelConfig offline_model_config;
  memset(&offline_model_config, 0, sizeof(offline_model_config));
  offline_model_config.debug = 0;
  offline_model_config.num_threads = 1;
  offline_model_config.provider = provider;
  offline_model_config.tokens = tokens_filename;
  offline_model_config.transducer.encoder = encoder_filename;
  offline_model_config.transducer.decoder = decoder_filename;
  offline_model_config.transducer.joiner = joiner_filename;

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

  printf("Recognized text: %s\n", result->text);

  SherpaOnnxDestroyOfflineRecognizerResult(result);
  SherpaOnnxDestroyOfflineStream(stream);
  SherpaOnnxDestroyOfflineRecognizer(recognizer);
  SherpaOnnxFreeWave(wave);

  return 0;
}
