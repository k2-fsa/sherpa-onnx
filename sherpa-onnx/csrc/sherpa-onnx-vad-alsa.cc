// sherpa-onnx/csrc/sherpa-onnx-vad-alsa.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

#include "sherpa-onnx/csrc/alsa.h"
#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"
#include "sherpa-onnx/csrc/wave-writer.h"

bool stop = false;
static void Handler(int32_t sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program shows how to use VAD in sherpa-onnx.

  ./bin/sherpa-onnx-vad-alsa \
    --silero-vad-model=/path/to/silero_vad.onnx \
    device_name

Please download silero_vad.onnx from
https://github.com/snakers4/silero-vad/blob/master/files/silero_vad.onnx

For instance, use
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx

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
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::VadModelConfig config;

  config.Register(&po);
  po.Read(argc, argv);
  if (po.NumArgs() != 1) {
    fprintf(stderr, "Please provide only 1 argument: the device name\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  std::string device_name = po.GetArg(1);
  sherpa_onnx::Alsa alsa(device_name.c_str());
  fprintf(stderr, "Use recording device: %s\n", device_name.c_str());

  int32_t sample_rate = 16000;

  if (alsa.GetExpectedSampleRate() != sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            sample_rate);
    exit(-1);
  }

  int32_t chunk = 0.1 * alsa.GetActualSampleRate();

  auto vad = std::make_unique<sherpa_onnx::VoiceActivityDetector>(config);

  fprintf(stderr, "Started. Please speak\n");

  int32_t window_size = config.silero_vad.window_size;
  bool printed = false;

  int32_t k = 0;
  while (!stop) {
    {
      const std::vector<float> &samples = alsa.Read(chunk);

      vad->AcceptWaveform(samples.data(), samples.size());

      if (vad->IsSpeechDetected() && !printed) {
        printed = true;
        fprintf(stderr, "\nDetected speech!\n");
      }
      if (!vad->IsSpeechDetected()) {
        printed = false;
      }

      while (!vad->Empty()) {
        const auto &segment = vad->Front();
        float duration =
            segment.samples.size() / static_cast<float>(sample_rate);

        fprintf(stderr, "Duration: %.3f seconds\n", duration);

        char filename[128];
        snprintf(filename, sizeof(filename), "seg-%d-%.3fs.wav", k, duration);
        k += 1;
        sherpa_onnx::WriteWave(filename, 16000, segment.samples.data(),
                               segment.samples.size());
        fprintf(stderr, "Saved to %s\n", filename);
        fprintf(stderr, "----------\n");

        vad->Pop();
      }
    }
  }

  return 0;
}
