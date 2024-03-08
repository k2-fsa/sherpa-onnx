// c-api-examples/asr-microphone-example/c-api-alsa.cc
// Copyright (c)  2022-2024  Xiaomi Corporation

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c-api-examples/asr-microphone-example/alsa.h"
#include "cargs.h"
#include "sherpa-onnx/c-api/c-api.h"

static struct cag_option options[] = {
    {.identifier = 'h',
     .access_letters = "h",
     .access_name = "help",
     .description = "Show help"},
    {.identifier = 't',
     .access_letters = NULL,
     .access_name = "tokens",
     .value_name = "tokens",
     .description = "Tokens file"},
    {.identifier = 'e',
     .access_letters = NULL,
     .access_name = "encoder",
     .value_name = "encoder",
     .description = "Encoder ONNX file"},
    {.identifier = 'd',
     .access_letters = NULL,
     .access_name = "decoder",
     .value_name = "decoder",
     .description = "Decoder ONNX file"},
    {.identifier = 'j',
     .access_letters = NULL,
     .access_name = "joiner",
     .value_name = "joiner",
     .description = "Joiner ONNX file"},
    {.identifier = 'n',
     .access_letters = NULL,
     .access_name = "num-threads",
     .value_name = "num-threads",
     .description = "Number of threads"},
    {.identifier = 'p',
     .access_letters = NULL,
     .access_name = "provider",
     .value_name = "provider",
     .description = "Provider: cpu (default), cuda, coreml"},
    {.identifier = 'm',
     .access_letters = NULL,
     .access_name = "decoding-method",
     .value_name = "decoding-method",
     .description =
         "Decoding method: greedy_search (default), modified_beam_search"},
    {.identifier = 'f',
     .access_letters = NULL,
     .access_name = "hotwords-file",
     .value_name = "hotwords-file",
     .description = "The file containing hotwords, one words/phrases per line, "
                    "and for each phrase the bpe/cjkchar are separated by a "
                    "space. For example: ▁HE LL O ▁WORLD, 你 好 世 界"},
    {.identifier = 's',
     .access_letters = NULL,
     .access_name = "hotwords-score",
     .value_name = "hotwords-score",
     .description = "The bonus score for each token in hotwords. Used only "
                    "when decoding_method is modified_beam_search"},
};

const char *kUsage =
    R"(
Usage:
  ./bin/c-api-alsa \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/decoder.onnx \
    device_name

The device name specifies which microphone to use in case there are several
on your system. You can use

  arecord -l

to find all available microphones on your computer. For instance, if it outputs

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and the device 0 on that card, please use:

  plughw:3,0

as the device_name.
)";

int main() {
  if (argc < 6) {
    fprintf(stderr, "%s\n", kUsage);
    exit(0);
  }

  SherpaOnnxOnlineRecognizerConfig config;
  memset(&config, 0, sizeof(config));

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
      case 't':
        config.model_config.tokens = value;
        break;
      case 'e':
        config.model_config.transducer.encoder = value;
        break;
      case 'd':
        config.model_config.transducer.decoder = value;
        break;
      case 'j':
        config.model_config.transducer.joiner = value;
        break;
      case 'n':
        config.model_config.num_threads = atoi(value);
        break;
      case 'p':
        config.model_config.provider = value;
        break;
      case 'm':
        config.decoding_method = value;
        break;
      case 'f':
        config.hotwords_file = value;
        break;
      case 's':
        config.hotwords_score = atof(value);
        break;
      case 'h': {
        fprintf(stderr, "%s\n", kUsage);
        exit(0);
        break;
      }
      default:
        // do nothing as config already has valid default values
        break;
    }
  }

  SherpaOnnxOnlineRecognizer *recognizer = CreateOnlineRecognizer(&config);
  SherpaOnnxOnlineStream *stream = CreateOnlineStream(recognizer);

  SherpaOnnxDisplay *display = CreateDisplay(50);
  int32_t segment_id = 0;

  const char *device_name = argv[context.index];
  fprintf(stderr, "device_name: %s\n", device_name);

  // free allocated resources
  DestroyDisplay(display);
  DestroyOnlineStream(stream);
  DestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
