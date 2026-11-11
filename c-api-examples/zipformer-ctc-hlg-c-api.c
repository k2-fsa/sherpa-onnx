// c-api-examples/zipformer-ctc-hlg-c-api.c
//
// Copyright (c)  2026  Marcin Baszczewski

//
// This file demonstrates how to decode a non-streaming CTC model with an
// HLG decoding graph using sherpa-onnx's C API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  const char *dir = "./sherpa-onnx-zipformer-ctc-en-2023-10-02";
  char wav_filename[256];
  char model_filename[256];
  char tokens_filename[256];
  char graph_filename[256];

  snprintf(wav_filename, sizeof(wav_filename), "%s/test_wavs/0.wav", dir);
  snprintf(model_filename, sizeof(model_filename), "%s/model.onnx", dir);
  snprintf(tokens_filename, sizeof(tokens_filename), "%s/tokens.txt", dir);

  // The model above also ships H.fst and HL.fst; any of the three works.
  snprintf(graph_filename, sizeof(graph_filename), "%s/HLG.fst", dir);

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
  recognizer_config.model_config.provider = "cpu";
  recognizer_config.model_config.tokens = tokens_filename;
  recognizer_config.model_config.zipformer_ctc.model = model_filename;

  recognizer_config.ctc_fst_decoder_config.graph = graph_filename;
  recognizer_config.ctc_fst_decoder_config.max_active = 3000;

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

  // The graph's output labels are in the json field, under "words".
  fprintf(stderr, "Result as json: %s\n", result->json);

  SherpaOnnxDestroyOfflineRecognizerResult(result);
  SherpaOnnxDestroyOfflineStream(stream);
  SherpaOnnxDestroyOfflineRecognizer(recognizer);
  SherpaOnnxFreeWave(wave);

  return 0;
}
