// sherpa-onnx/csrc/sherpa-onnx-alsa-offline.cc
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cctype>  // std::tolower
#include <chrono>  // NOLINT
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "sherpa-onnx/csrc/alsa.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"

enum class State {
  kIdle,
  kRecording,
  kDecoding,
};

State state = State::kIdle;

// true to stop the program and exit
bool stop = false;

std::vector<float> samples;
std::mutex samples_mutex;

static void DetectKeyPress() {
  SHERPA_ONNX_LOGE("Press Enter to start");
  int32_t key;
  while (!stop && (key = getchar())) {
    if (key != 0x0a) {
      continue;
    }

    switch (state) {
      case State::kIdle:
        SHERPA_ONNX_LOGE("Start recording. Press Enter to stop recording");
        state = State::kRecording;
        {
          std::lock_guard<std::mutex> lock(samples_mutex);
          samples.clear();
        }
        break;
      case State::kRecording:
        SHERPA_ONNX_LOGE("Stop recording. Decoding ...");
        state = State::kDecoding;
        break;
      case State::kDecoding:
        break;
    }
  }
}

static void Record(const char *device_name, int32_t expected_sample_rate) {
  sherpa_onnx::Alsa alsa(device_name);

  if (alsa.GetExpectedSampleRate() != expected_sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            expected_sample_rate);
    exit(-1);
  }

  int32_t chunk = 0.1 * alsa.GetActualSampleRate();
  while (!stop) {
    std::lock_guard<std::mutex> lock(samples_mutex);
    const std::vector<float> &s = alsa.Read(chunk);
    samples.insert(samples.end(), s.begin(), s.end());
  }
}

static void Handler(int32_t sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Press Enter to exit\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program uses non-streaming models with microphone for speech recognition.
Usage:

(1) Transducer from icefall

  ./bin/sherpa-onnx-alsa-offline \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    device_name

(2) Paraformer from FunASR

  ./bin/sherpa-onnx-alsa-offline \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    --num-threads=1 \
    device_name

(3) Whisper models

  ./bin/sherpa-onnx-alsa-offline \
    --whisper-encoder=./sherpa-onnx-whisper-base.en/base.en-encoder.int8.onnx \
    --whisper-decoder=./sherpa-onnx-whisper-base.en/base.en-decoder.int8.onnx \
    --tokens=./sherpa-onnx-whisper-base.en/base.en-tokens.txt \
    --num-threads=1 \
    device_name

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

  plughw:3,0

as the device_name.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OfflineRecognizerConfig config;
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

  SHERPA_ONNX_LOGE("Creating recognizer ...");
  sherpa_onnx::OfflineRecognizer recognizer(config);
  SHERPA_ONNX_LOGE("Recognizer created!");

  std::string device_name = po.GetArg(1);
  fprintf(stderr, "Use recording device: %s\n", device_name.c_str());

  int32_t sample_rate = config.feat_config.sampling_rate;

  std::thread t(DetectKeyPress);
  std::thread t2(Record, device_name.c_str(), sample_rate);

  while (!stop) {
    switch (state) {
      case State::kIdle:
        break;
      case State::kRecording:
        break;
      case State::kDecoding: {
        std::vector<float> buf;
        {
          std::lock_guard<std::mutex> lock(samples_mutex);
          buf = std::move(samples);
        }

        auto s = recognizer.CreateStream();
        s->AcceptWaveform(sample_rate, buf.data(), buf.size());
        recognizer.DecodeStream(s.get());
        SHERPA_ONNX_LOGE("Decoding Done! Result is:");
        SHERPA_ONNX_LOGE("%s", s->GetResult().text.c_str());

        state = State::kIdle;
        SHERPA_ONNX_LOGE("Press Enter to start");
        break;
      }
    }

    using namespace std::chrono_literals;  // NOLINT
    std::this_thread::sleep_for(20ms);     // sleep for 20ms
  }
  t.join();
  t2.join();

  return 0;
}
