// cxx-api-examples/fire-red-asr-ctc-simulate-streaming-alsa-cxx-api.cc
// Copyright (c)  2025  Xiaomi Corporation

//
// This file demonstrates how to use FireRedASR CTC models with sherpa-onnx's
// C++ API for streaming speech recognition from a microphone.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
// tar xvf sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
// rm sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
//
// clang-format on

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <iostream>
#include <mutex>  // NOLINT
#include <queue>
#include <thread>  // NOLINT
#include <vector>

#include "sherpa-display.h"  // NOLINT
#include "sherpa-onnx/c-api/cxx-api.h"
#include "sherpa-onnx/csrc/alsa.h"

std::queue<std::vector<float>> samples_queue;
std::condition_variable condition_variable;
std::mutex mutex;
bool stop = false;

static void Handler(int32_t /*sig*/) {
  stop = true;
  condition_variable.notify_one();
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

static void RecordCallback(sherpa_onnx::Alsa *alsa) {
  int32_t chunk = 0.1 * alsa->GetActualSampleRate();
  while (!stop) {
    std::vector<float> samples = alsa->Read(chunk);

    std::lock_guard<std::mutex> lock(mutex);
    samples_queue.emplace(std::move(samples));
    condition_variable.notify_one();
  }
}

static sherpa_onnx::cxx::VoiceActivityDetector CreateVad() {
  using namespace sherpa_onnx::cxx;  // NOLINT
  VadModelConfig config;
  config.silero_vad.model = "./silero_vad.onnx";
  config.silero_vad.threshold = 0.5;
  config.silero_vad.min_silence_duration = 0.1;
  config.silero_vad.min_speech_duration = 0.25;
  config.silero_vad.max_speech_duration = 8;
  config.sample_rate = 16000;
  config.debug = false;

  VoiceActivityDetector vad = VoiceActivityDetector::Create(config, 20);
  if (!vad.Get()) {
    std::cerr << "Failed to create VAD. Please check your config\n";
    exit(-1);
  }

  return vad;
}

static sherpa_onnx::cxx::OfflineRecognizer CreateOfflineRecognizer() {
  using namespace sherpa_onnx::cxx;  // NOLINT
  OfflineRecognizerConfig config;

  config.model_config.fire_red_asr_ctc.model =
      "./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx";
  config.model_config.tokens =
      "./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt";

  config.model_config.num_threads = 2;
  config.model_config.debug = false;

  std::cout << "Loading model\n";
  OfflineRecognizer recognizer = OfflineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    exit(-1);
  }
  std::cout << "Loading model done\n";
  return recognizer;
}

int32_t main(int32_t argc, const char *argv[]) {
  const char *kUsageMessage = R"usage(
Usage:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
tar xvf sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
rm sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2

./fire-red-asr-ctc-simulate-streaming-alsa-cxx-api device_name

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

  if (argc != 2) {
    fprintf(stderr, "%s\n", kUsageMessage);
    return -1;
  }

  signal(SIGINT, Handler);

  using namespace sherpa_onnx::cxx;  // NOLINT

  auto vad = CreateVad();
  auto recognizer = CreateOfflineRecognizer();

  int32_t expected_sample_rate = 16000;

  std::string device_name = argv[1];
  sherpa_onnx::Alsa alsa(device_name.c_str());
  fprintf(stderr, "Use recording device: %s\n", device_name.c_str());

  if (alsa.GetExpectedSampleRate() != expected_sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            expected_sample_rate);
    exit(-1);
  }

  int32_t window_size = 512;  // samples, please don't change

  int32_t offset = 0;
  std::vector<float> buffer;
  bool speech_started = false;

  auto started_time = std::chrono::steady_clock::now();

  SherpaDisplay display;

  std::thread record_thread(RecordCallback, &alsa);

  std::cout << "Started! Please speak\n";

  while (!stop) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      while (samples_queue.empty() && !stop) {
        condition_variable.wait(lock);
      }
      if (stop) {
        break;
      }

      const auto &s = samples_queue.front();
      buffer.insert(buffer.end(), s.begin(), s.end());

      samples_queue.pop();
    }

    for (; offset + window_size < buffer.size(); offset += window_size) {
      vad.AcceptWaveform(buffer.data() + offset, window_size);
      if (!speech_started && vad.IsDetected()) {
        speech_started = true;
        started_time = std::chrono::steady_clock::now();
      }
    }
    if (!speech_started) {
      if (buffer.size() > 10 * window_size) {
        offset -= buffer.size() - 10 * window_size;
        buffer = {buffer.end() - 10 * window_size, buffer.end()};
      }
    }

    auto current_time = std::chrono::steady_clock::now();
    const float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time -
                                                              started_time)
            .count() /
        1000.;

    if (speech_started && elapsed_seconds > 0.2) {
      OfflineStream stream = recognizer.CreateStream();
      stream.AcceptWaveform(expected_sample_rate, buffer.data(), buffer.size());

      recognizer.Decode(&stream);

      OfflineRecognizerResult result = recognizer.GetResult(&stream);
      display.UpdateText(result.text);
      display.Display();

      started_time = std::chrono::steady_clock::now();
    }

    while (!vad.IsEmpty()) {
      auto segment = vad.Front();

      vad.Pop();

      OfflineStream stream = recognizer.CreateStream();
      stream.AcceptWaveform(expected_sample_rate, segment.samples.data(),
                            segment.samples.size());

      recognizer.Decode(&stream);

      OfflineRecognizerResult result = recognizer.GetResult(&stream);

      display.UpdateText(result.text);
      display.FinalizeCurrentSentence();
      display.Display();

      buffer.clear();
      offset = 0;
      speech_started = false;
    }
  }

  record_thread.join();

  return 0;
}
