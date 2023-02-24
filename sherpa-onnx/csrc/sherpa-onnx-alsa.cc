// sherpa-onnx/csrc/sherpa-onnx-alsa.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cctype>  // std::tolower
#include <cstdint>

#include "sherpa-onnx/csrc/alsa.h"
#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/online-recognizer.h"

bool stop = false;

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int main(int32_t argc, char *argv[]) {
  if (argc < 6 || argc > 7) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-onnx-alsa \
    /path/to/tokens.txt \
    /path/to/encoder.onnx \
    /path/to/decoder.onnx \
    /path/to/joiner.onnx \
    device_name \
    [num_threads]

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.

The device name specifies which microphone to use in case there are several
on you system. You can use

  arecord -l

to find all available microphones on your computer. For instance, if it outputs

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and the device 0 on that card, please use:

  hw:3,0

as the device_name.
)usage";

    fprintf(stderr, "%s\n", usage);
    fprintf(stderr, "argc, %d\n", argc);

    return 0;
  }

  signal(SIGINT, Handler);

  sherpa_onnx::OnlineRecognizerConfig config;

  config.model_config.tokens = argv[1];

  config.model_config.debug = false;
  config.model_config.encoder_filename = argv[2];
  config.model_config.decoder_filename = argv[3];
  config.model_config.joiner_filename = argv[4];

  const char *device_name = argv[5];

  config.model_config.num_threads = 2;
  if (argc == 7 && atoi(argv[6]) > 0) {
    config.model_config.num_threads = atoi(argv[6]);
  }

  config.enable_endpoint = true;

  config.endpoint_config.rule1.min_trailing_silence = 2.4;
  config.endpoint_config.rule2.min_trailing_silence = 1.2;
  config.endpoint_config.rule3.min_utterance_length = 300;

  fprintf(stderr, "%s\n", config.ToString().c_str());

  sherpa_onnx::OnlineRecognizer recognizer(config);

  int32_t expected_sample_rate = config.feat_config.sampling_rate;

  sherpa_onnx::Alsa alsa(device_name);
  fprintf(stderr, "Use recording device: %s\n", device_name);

  if (alsa.GetExpectedSampleRate() != expected_sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            expected_sample_rate);
    exit(-1);
  }

  int32_t chunk = 0.1 * alsa.GetActualSampleRate();

  std::string last_text;

  auto stream = recognizer.CreateStream();

  sherpa_onnx::Display display;

  int32_t segment_index = 0;
  while (!stop) {
    const std::vector<float> samples = alsa.Read(chunk);

    stream->AcceptWaveform(expected_sample_rate, samples.data(),
                           samples.size());

    while (recognizer.IsReady(stream.get())) {
      recognizer.DecodeStream(stream.get());
    }

    auto text = recognizer.GetResult(stream.get()).text;

    bool is_endpoint = recognizer.IsEndpoint(stream.get());

    if (!text.empty() && last_text != text) {
      last_text = text;

      std::transform(text.begin(), text.end(), text.begin(),
                     [](auto c) { return std::tolower(c); });

      display.Print(segment_index, text);
    }

    if (!text.empty() && is_endpoint) {
      ++segment_index;
      recognizer.Reset(stream.get());
    }
  }

  return 0;
}
