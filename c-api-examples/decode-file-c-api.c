// c-api-examples/decode-file-c-api.c
//
// Copyright (c)  2023  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// to decode a file.

#include "cargs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

static struct cag_option options[] = {
  {
    .identifier = 't',
    .access_letters = NULL,
    .access_name = "tokens",
    .value_name = "tokens",
    .description = "Tokens file"
  }, {
    .identifier = 'e',
    .access_letters = NULL,
    .access_name = "encoder",
    .value_name = "encoder",
    .description = "Encoder ONNX file"
  }, {
    .identifier = 'd',
    .access_letters = NULL,
    .access_name = "decoder",
    .value_name = "decoder",
    .description = "Decoder ONNX file"
  }, {
    .identifier = 'j',
    .access_letters = NULL,
    .access_name = "joiner",
    .value_name = "joiner",
    .description = "Joiner ONNX file"
  }, {
    .identifier = 'n',
    .access_letters = NULL,
    .access_name = "num-threads",
    .value_name = "num-threads",
    .description = "Number of threads"
  }, {
    .identifier = 'p',
    .access_letters = NULL,
    .access_name = "provider",
    .value_name = "provider",
    .description = "Provider: cpu (default), cuda, coreml"
  }, {
    .identifier = 'm',
    .access_letters = NULL,
    .access_name = "decoding-method",
    .value_name = "decoding-method",
    .description = 
      "Decoding method: greedy_search (default), modified_beam_search"
  }
};

const char *kUsage =
    "\n"
    "Usage:\n "
    "  ./bin/decode-file-c-api \\\n"
    "    --tokens=/path/to/tokens.txt \\\n"
    "    --encoder=/path/to/encoder.onnx \\\n"
    "    --decoder=/path/to/decoder.onnx \\\n"
    "    --joiner=/path/to/joiner.onnx \\\n"
    "    /path/to/foo.wav\n"
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

  cag_option_context context;
  char identifier;
  const char *value;

  cag_option_prepare(&context, options, CAG_ARRAY_SIZE(options), argc, argv);

  while (cag_option_fetch(&context)) {
    identifier = cag_option_get(&context);
    value = cag_option_get_value(&context);
    switch (identifier) {
      case 't': config.model_config.tokens = value; break;
      case 'e': config.model_config.encoder = value; break;
      case 'd': config.model_config.decoder = value; break;
      case 'j': config.model_config.joiner = value; break;
      case 'n': config.model_config.num_threads = atoi(value); break;
      case 'p': config.model_config.provider = value; break;
      case 'm': config.decoding_method = value; break;
      default: 
        // do nothing as config already have valid default values
        break;
    }
  }
  
  SherpaOnnxOnlineRecognizer *recognizer = CreateOnlineRecognizer(&config);
  SherpaOnnxOnlineStream *stream = CreateOnlineStream(recognizer);

  SherpaOnnxDisplay *display = CreateDisplay(50);
  int32_t segment_id = 0;

  const char *wav_filename = argv[context.index];
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
