// c-api-examples/streaming-paraformer-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation

//
// This file demonstrates how to use streaming Paraformer with sherpa-onnx's C
// API.
// clang-format off
// 
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
// tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
// rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  const char *wav_filename =
      "sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/0.wav";
  const char *encoder_filename =
      "sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx";
  const char *decoder_filename =
      "sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx";
  const char *tokens_filename =
      "sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt";
  const char *provider = "cpu";

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  // Paraformer config
  SherpaOnnxOnlineParaformerModelConfig paraformer_config;
  memset(&paraformer_config, 0, sizeof(paraformer_config));
  paraformer_config.encoder = encoder_filename;
  paraformer_config.decoder = decoder_filename;

  // Online model config
  SherpaOnnxOnlineModelConfig online_model_config;
  memset(&online_model_config, 0, sizeof(online_model_config));
  online_model_config.debug = 1;
  online_model_config.num_threads = 1;
  online_model_config.provider = provider;
  online_model_config.tokens = tokens_filename;
  online_model_config.paraformer = paraformer_config;

  // Recognizer config
  SherpaOnnxOnlineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "greedy_search";
  recognizer_config.model_config = online_model_config;

  const SherpaOnnxOnlineRecognizer *recognizer =
      SherpaOnnxCreateOnlineRecognizer(&recognizer_config);

  if (recognizer == NULL) {
    fprintf(stderr, "Please check your config!\n");
    SherpaOnnxFreeWave(wave);
    return -1;
  }

  const SherpaOnnxOnlineStream *stream =
      SherpaOnnxCreateOnlineStream(recognizer);

  const SherpaOnnxDisplay *display = SherpaOnnxCreateDisplay(50);
  int32_t segment_id = 0;

// simulate streaming. You can choose an arbitrary N
#define N 3200

  fprintf(stderr, "sample rate: %d, num samples: %d, duration: %.2f s\n",
          wave->sample_rate, wave->num_samples,
          (float)wave->num_samples / wave->sample_rate);

  int32_t k = 0;
  while (k < wave->num_samples) {
    int32_t start = k;
    int32_t end =
        (start + N > wave->num_samples) ? wave->num_samples : (start + N);
    k += N;

    SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate,
                                         wave->samples + start, end - start);
    while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
      SherpaOnnxDecodeOnlineStream(recognizer, stream);
    }

    const SherpaOnnxOnlineRecognizerResult *r =
        SherpaOnnxGetOnlineStreamResult(recognizer, stream);

    if (strlen(r->text)) {
      SherpaOnnxPrint(display, segment_id, r->text);
    }

    if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream)) {
      if (strlen(r->text)) {
        ++segment_id;
      }
      SherpaOnnxOnlineStreamReset(recognizer, stream);
    }

    SherpaOnnxDestroyOnlineRecognizerResult(r);
  }

  // add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail_paddings,
                                       4800);

  SherpaOnnxFreeWave(wave);

  SherpaOnnxOnlineStreamInputFinished(stream);
  while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
    SherpaOnnxDecodeOnlineStream(recognizer, stream);
  }

  const SherpaOnnxOnlineRecognizerResult *r =
      SherpaOnnxGetOnlineStreamResult(recognizer, stream);

  if (strlen(r->text)) {
    SherpaOnnxPrint(display, segment_id, r->text);
  }

  SherpaOnnxDestroyOnlineRecognizerResult(r);

  SherpaOnnxDestroyDisplay(display);
  SherpaOnnxDestroyOnlineStream(stream);
  SherpaOnnxDestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
