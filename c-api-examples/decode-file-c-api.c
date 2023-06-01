// c-api-examples/decode-file-c-api.c
//
// Copyright (c)  2023  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// to decode a file.

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sherpa-onnx/c-api/c-api.h"

const char *kUsage =
    "\n"
    "Usage:\n "
    "  ./bin/decode-file-c-api \\\n"
    "    /path/to/tokens.txt \\\n"
    "    /path/to/encoder.onnx \\\n"
    "    /path/to/decoder.onnx \\\n"
    "    /path/to/joiner.onnx \\\n"
    "    /path/to/foo.wav [num_threads [decoding_method] [provider]]\n"
    "\n\n"
    "Default num_threads is 1.\n"
    "Valid decoding_method: greedy_search (default), modified_beam_search\n\n"
    "Valid provider: cpu (default), cuda, coreml\n\n"
    "Please refer to \n"
    "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html\n"
    "for a list of pre-trained models to download.\n";

int32_t main(int32_t argc, char *argv[]) {
  SherpaOnnxOnlineRecognizerConfig config;

  config.model_config.debug = 0;
  config.model_config.num_threads = 1;
  config.model_config.provider = "cpu";

  config.decoding_method = "greedy_search";

  config.max_active_paths = 4;

  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;

  config.enable_endpoint = 1;
  config.rule1_min_trailing_silence = 2.4;
  config.rule2_min_trailing_silence = 1.2;
  config.rule3_min_utterance_length = 300;

  int opt;
      
  struct option long_options[] = {
    {"tokens", required_argument, 0, 't'},
    {"encoder", required_argument, 0, 'e'},
    {"decoder", required_argument, 0, 'd'},
    {"joiner", required_argument, 0, 'j'},
    {"num-threads", required_argument, 0, 'n'},
    {"provider", required_argument, 0, 'p'},
    {"decoding-method", required_argument, 0, 'm'},
    {0, 0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
    switch (opt) {
      case 't': config.model_config.tokens = optarg; break;
      case 'e': config.model_config.encoder = optarg; break;
      case 'd': config.model_config.decoder = optarg; break;
      case 'j': config.model_config.joiner = optarg; break;
      case 'n': config.model_config.num_threads = optarg; break;
      case 'p': config.model_config.provider = optarg; break;
      case 'm': config.decoding_method = optarg; break;
      case '?':
        fprintf(stderr, "Invalid option: -%c\n", optopt);
        fprintf(stderr, "%s\n", kUsage);
        return EXIT_FAILURE;
      case ':':
        fprintf(stderr, "Invalid option: -%c requires an argument\n", optopt);
        fprintf(stderr, "%s\n", kUsage);
        return EXIT_FAILURE;
    }
  }

  SherpaOnnxOnlineRecognizer *recognizer = CreateOnlineRecognizer(&config);
  SherpaOnnxOnlineStream *stream = CreateOnlineStream(recognizer);

  SherpaOnnxDisplay *display = CreateDisplay(50);
  int32_t segment_id = 0;

  const char *wav_filename = argv[optind];
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
  DestroyOnlineStream(stream);
  DestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
