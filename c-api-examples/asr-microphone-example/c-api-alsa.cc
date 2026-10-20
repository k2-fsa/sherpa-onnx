// c-api-examples/asr-microphone-example/c-api-alsa.cc
// Copyright (c)  2022-2024  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// for real-time speech recognition from a microphone.
//
// It runs only on Linux.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
tar xf sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
rm sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2

./c-api-alsa

 */
// clang-format on

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cctype>  // std::tolower
#include <cstdint>
#include <string>
#include <vector>

#include "c-api-examples/asr-microphone-example/alsa.h"
#include "sherpa-onnx/c-api/c-api.h"

bool stop = false;

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main() {
  // Recording device to use. Run
  //
  //   arecord -l
  //
  // to list all available microphones. For instance, if it prints
  //
  // **** List of CAPTURE Hardware Devices ****
  // card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  //
  // and you want card 3, device 0, then use "plughw:3,0" below.
  const char *device_name = "default";

  const char *tokens =
      "./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/"
      "tokens.txt";
  const char *encoder =
      "./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/"
      "encoder.onnx";
  const char *decoder =
      "./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/"
      "decoder.onnx";
  const char *joiner =
      "./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/"
      "joiner.onnx";

  signal(SIGINT, Handler);

  SherpaOnnxOnlineRecognizerConfig config;
  memset(&config, 0, sizeof(config));

  config.model_config.debug = 0;
  config.model_config.num_threads = 1;
  config.model_config.provider = "cpu";
  config.model_config.tokens = tokens;
  config.model_config.transducer.encoder = encoder;
  config.model_config.transducer.decoder = decoder;
  config.model_config.transducer.joiner = joiner;

  config.decoding_method = "greedy_search";

  config.max_active_paths = 4;

  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;

  config.enable_endpoint = 1;
  config.rule1_min_trailing_silence = 2.4;
  config.rule2_min_trailing_silence = 1.2;
  config.rule3_min_utterance_length = 300;

  const SherpaOnnxOnlineRecognizer *recognizer =
      SherpaOnnxCreateOnlineRecognizer(&config);
  const SherpaOnnxOnlineStream *stream =
      SherpaOnnxCreateOnlineStream(recognizer);

  const SherpaOnnxDisplay *display = SherpaOnnxCreateDisplay(50);
  int32_t segment_id = 0;

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
    SherpaOnnxOnlineStreamAcceptWaveform(stream, expected_sample_rate,
                                         samples.data(), samples.size());
    while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
      SherpaOnnxDecodeOnlineStream(recognizer, stream);
    }

    const SherpaOnnxOnlineRecognizerResult *r =
        SherpaOnnxGetOnlineStreamResult(recognizer, stream);

    std::string text = r->text;
    SherpaOnnxDestroyOnlineRecognizerResult(r);

    if (!text.empty() && last_text != text) {
      last_text = text;

      std::transform(text.begin(), text.end(), text.begin(),
                     [](auto c) { return std::tolower(c); });

      SherpaOnnxPrint(display, segment_index, text.c_str());
      fflush(stderr);
    }

    if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream)) {
      if (!text.empty()) {
        ++segment_index;
      }
      SherpaOnnxOnlineStreamReset(recognizer, stream);
    }
  }

  // free allocated resources
  SherpaOnnxDestroyDisplay(display);
  SherpaOnnxDestroyOnlineStream(stream);
  SherpaOnnxDestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
