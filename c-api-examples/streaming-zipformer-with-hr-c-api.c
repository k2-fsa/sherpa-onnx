// c-api-examples/streaming-zipformer-with-hr-c-api.c
//
// Copyright (c)  2025  Xiaomi Corporation

//
// This file demonstrates how to use streaming Zipformer with sherpa-onnx's C
// API.
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
// tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
// rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
// tar xf dict.tar.bz2
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  const char *wav_filename = "test-hr.wav";

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  // Online model config
  SherpaOnnxOnlineModelConfig online_model_config;
  memset(&online_model_config, 0, sizeof(online_model_config));
  online_model_config.debug = 0;
  online_model_config.num_threads = 1;
  online_model_config.provider = "cpu";
  online_model_config.tokens =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt";

  online_model_config.transducer.encoder =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
      "encoder-epoch-99-avg-1.int8.onnx";

  // Note: We recommend not using int8.onnx for the decoder.
  online_model_config.transducer.decoder =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
      "decoder-epoch-99-avg-1.onnx";

  online_model_config.transducer.joiner =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
      "joiner-epoch-99-avg-1.int8.onnx";

  online_model_config.tokens =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt";

  online_model_config.num_threads = 1;

  // Recognizer config
  SherpaOnnxOnlineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "greedy_search";
  recognizer_config.model_config = online_model_config;

  recognizer_config.hr.dict_dir = "./dict";
  recognizer_config.hr.lexicon = "./lexicon.txt";

  // Please see
  // https://colab.research.google.com/drive/1jEaS3s8FbRJIcVQJv2EQx19EM_mnuARi?usp=sharing
  // for how to generate your own replace.fst
  recognizer_config.hr.rule_fsts = "./replace.fst";

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
