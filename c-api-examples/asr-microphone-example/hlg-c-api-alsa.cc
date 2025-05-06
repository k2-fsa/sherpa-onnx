// c-api-examples/asr-microphone-example/c-api-alsa.cc
// Copyright (c)  2022-2024  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cctype>  // std::tolower
#include <cstdint>
#include <string>

#include "c-api-examples/asr-microphone-example/alsa.h"
#include "sherpa-onnx/c-api/c-api.h"

bool stop = false;

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  if (argc != 2) {
    fprintf(stderr, R"(Please provide the device name.

The device name specifies which microphone to use in case there are several
on your system. You can use

  arecord -l

to find all available microphones on your computer. For instance, if it outputs

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and device 0 on that card, please use:

  plughw:3,0

as the device_name.
)");
  return -1;
  }

  // clang-format off
  //
  // Please download the model from
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  const char *model = "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx";
  const char *tokens = "./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt";
  const char *graph = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst";
  graph = "";
  // clang-format on

  SherpaOnnxOnlineRecognizerConfig config;

  memset(&config, 0, sizeof(config));
  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;
  config.model_config.zipformer2_ctc.model = model;
  config.model_config.tokens = tokens;
  config.model_config.num_threads = 1;
  config.model_config.provider = "cpu";
  config.model_config.debug = 0;
  config.ctc_fst_decoder_config.graph = graph;

  config.enable_endpoint = 1;
  config.rule1_min_trailing_silence = 2.4;
  config.rule2_min_trailing_silence = 1.2;
  config.rule3_min_utterance_length = 300;

  const SherpaOnnxOnlineRecognizer *recognizer =
      CreateOnlineRecognizer(&config);
  if (!recognizer) {
    fprintf(stderr, "Failed to create recognizer");
    exit(-1);
  }

  const SherpaOnnxOnlineStream *stream = CreateOnlineStream(recognizer);

  const SherpaOnnxDisplay *display = CreateDisplay(50);
  int32_t segment_id = 0;

  // please use arecord -l to find your device
  const char *device_name = argv[1];
  sherpa_onnx::Alsa alsa(device_name);
  fprintf(stderr, "Use recording device: %s\n", device_name);
  fprintf(stderr,
          "Please \033[32m\033[1mspeak\033[0m! Press \033[31m\033[1mCtrl + "
          "C\033[0m to exit\n");

  int32_t expected_sample_rate = 16000;

  if (alsa.GetExpectedSampleRate() != expected_sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            expected_sample_rate);
    exit(-1);
  }

  int32_t chunk = 0.1 * alsa.GetActualSampleRate();

  std::string last_text;

  int32_t segment_index = 0;

  while (!stop) {
    const std::vector<float> &samples = alsa.Read(chunk);
    AcceptWaveform(stream, expected_sample_rate, samples.data(),
                   samples.size());
    while (IsOnlineStreamReady(recognizer, stream)) {
      DecodeOnlineStream(recognizer, stream);
    }

    const SherpaOnnxOnlineRecognizerResult *r =
        GetOnlineStreamResult(recognizer, stream);

    std::string text = r->text;
    DestroyOnlineRecognizerResult(r);

    if (!text.empty() && last_text != text) {
      last_text = text;

      std::transform(text.begin(), text.end(), text.begin(),
                     [](auto c) { return std::tolower(c); });

      SherpaOnnxPrint(display, segment_index, text.c_str());
      fflush(stderr);
    }

    if (IsEndpoint(recognizer, stream)) {
      if (!text.empty()) {
        ++segment_index;
      }
      Reset(recognizer, stream);
    }
  }

  // free allocated resources
  DestroyDisplay(display);
  DestroyOnlineStream(stream);
  DestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
