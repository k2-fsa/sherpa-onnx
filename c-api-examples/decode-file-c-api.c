// c-api-examples/decode-file-c-api.c
//
// Copyright (c)  2023  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// to decode a file.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

const char *kUsage =
    "\n"
    "Usage:\n "
    "  ./bin/decode-file-c-api \\\n"
    "    /path/to/tokens.txt \\\n"
    "    /path/to/encoder.onnx \\\n"
    "    /path/to/decoder.onnx \\\n"
    "    /path/to/joiner.onnx \\\n"
    "    /path/to/foo.wav [num_threads [decoding_method]]\n"
    "\n\n"
    "Default num_threads is 1.\n"
    "Valid decoding_method: greedy_search (default), modified_beam_search\n\n"
    "Please refer to \n"
    "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html\n"
    "for a list of pre-trained models to download.\n";

int32_t main(int32_t argc, char *argv[]) {
  if (argc < 6 || argc > 8) {
    fprintf(stderr, "%s\n", kUsage);
    return -1;
  }
  SherpaOnnxOnlineRecognizerConfig config;
  config.model_config.tokens = argv[1];
  config.model_config.encoder = argv[2];
  config.model_config.decoder = argv[3];
  config.model_config.joiner = argv[4];

  int32_t num_threads = 1;
  if (argc == 7 && atoi(argv[6]) > 0) {
    num_threads = atoi(argv[6]);
  }
  config.model_config.num_threads = num_threads;
  config.model_config.debug = 0;

  config.decoding_method = "greedy_search";
  if (argc == 8) {
    config.decoding_method = argv[7];
  }

  config.max_active_paths = 4;

  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;

  config.enable_endpoint = 1;
  config.rule1_min_trailing_silence = 2.4;
  config.rule2_min_trailing_silence = 1.2;
  config.rule3_min_utterance_length = 300;

  SherpaOnnxOnlineRecognizer *recognizer = CreateOnlineRecognizer(&config);
  SherpaOnnxOnlineStream *stream = CreateOnlineStream(recognizer);

  SherpaOnnxDisplay *display = CreateDisplay(50);
  int32_t segment_id = 0;

  const char *wav_filename = argv[5];
  FILE *fp = fopen(wav_filename, "rb");
  if (!fp) {
    fprintf(stderr, "Failed to open %s\n", wav_filename);
    return -1;
  }

  // Assume the wave header occupies 44 bytes.
  fseek(fp, 44, SEEK_SET);

  // simulate streaming

#define N 3200  // 0.2 s. Sample rate is fixed to 16 kHz

  int16_t buffer[N];
  float samples[N];

  while (!feof(fp)) {
    size_t n = fread((void *)buffer, sizeof(int16_t), N, fp);
    if (n > 0) {
      for (size_t i = 0; i != n; ++i) {
        samples[i] = buffer[i] / 32768.;
      }
      AcceptWaveform(stream, 16000, samples, n);
      while (IsOnlineStreamReady(recognizer, stream)) {
        DecodeOnlineStream(recognizer, stream);
      }

      SherpaOnnxOnlineRecognizerResult *r =
          GetOnlineStreamResult(recognizer, stream);

      if (strlen(r->text)) {
        SherpaOnnxPrint(display, segment_id, r->text);
      }

      if (IsEndpoint(recognizer, stream)) {
        if (strlen(r->text)) {
          ++segment_id;
        }
        Reset(recognizer, stream);
      }

      DestroyOnlineRecognizerResult(r);
    }
  }
  fclose(fp);

  // add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  AcceptWaveform(stream, 16000, tail_paddings, 4800);

  InputFinished(stream);
  while (IsOnlineStreamReady(recognizer, stream)) {
    DecodeOnlineStream(recognizer, stream);
  }

  SherpaOnnxOnlineRecognizerResult *r =
      GetOnlineStreamResult(recognizer, stream);

  if (strlen(r->text)) {
    SherpaOnnxPrint(display, segment_id, r->text);
  }

  DestroyOnlineRecognizerResult(r);

  DestroyDisplay(display);
  DestoryOnlineStream(stream);
  DestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
