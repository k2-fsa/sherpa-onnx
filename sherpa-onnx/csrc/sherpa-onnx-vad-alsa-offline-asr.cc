// sherpa-onnx/csrc/sherpa-onnx-vad-alsa-offline-asr.cc
//
// Copyright (c)  2022-2025  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <mutex>  // NOLINT

#include "sherpa-onnx/csrc/alsa.h"
#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"

bool stop = false;
static void Handler(int32_t /*sig*/) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program shows how to use a streaming VAD with non-streaming ASR in
sherpa-onnx.

Please download silero_vad.onnx from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

For instance, use
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

Please refer to ./sherpa-onnx-microphone-offline.cc
to download models for offline ASR.

(1) Transducer from icefall

  ./bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    device_name

(2) Paraformer from FunASR

  ./bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    device_name

(3) Whisper models

  ./bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --whisper-encoder=./sherpa-onnx-whisper-base.en/base.en-encoder.int8.onnx \
    --whisper-decoder=./sherpa-onnx-whisper-base.en/base.en-decoder.int8.onnx \
    --tokens=./sherpa-onnx-whisper-base.en/base.en-tokens.txt \
    device_name

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
  sherpa_onnx::VadModelConfig vad_config;

  sherpa_onnx::OfflineRecognizerConfig asr_config;

  vad_config.Register(&po);
  asr_config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 1) {
    fprintf(stderr, "Please provide only 1 argument: the device name\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", vad_config.ToString().c_str());
  fprintf(stderr, "%s\n", asr_config.ToString().c_str());

  if (!vad_config.Validate()) {
    fprintf(stderr, "Errors in vad_config!\n");
    return -1;
  }

  if (!asr_config.Validate()) {
    fprintf(stderr, "Errors in asr_config!\n");
    return -1;
  }

  fprintf(stderr, "Creating recognizer ...\n");
  sherpa_onnx::OfflineRecognizer recognizer(asr_config);
  fprintf(stderr, "Recognizer created!\n");

  auto vad = std::make_unique<sherpa_onnx::VoiceActivityDetector>(vad_config);

  std::string device_name = po.GetArg(1);
  sherpa_onnx::Alsa alsa(device_name.c_str());
  fprintf(stderr, "Use recording device: %s\n", device_name.c_str());

  int32_t sample_rate = 16000;

  if (alsa.GetExpectedSampleRate() != sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            sample_rate);
    exit(-1);
  }

  fprintf(stderr, "Started. Please speak\n");

  int32_t window_size = vad_config.silero_vad.window_size;
  int32_t index = 0;

  while (!stop) {
    const std::vector<float> &samples = alsa.Read(window_size);
    vad->AcceptWaveform(samples.data(), samples.size());

    while (!vad->Empty()) {
      const auto &segment = vad->Front();
      auto s = recognizer.CreateStream();
      s->AcceptWaveform(sample_rate, segment.samples.data(),
                        segment.samples.size());
      recognizer.DecodeStream(s.get());
      const auto &result = s->GetResult();
      if (!result.text.empty()) {
        fprintf(stderr, "%2d: %s\n", index, result.text.c_str());
        ++index;
      }
      vad->Pop();
    }
  }

  return 0;
}
