// cxx-api-examples/sense-voice-simulate-streaming-microphone-cxx-api.cc
// Copyright (c)  2025  Xiaomi Corporation

//
// This file demonstrates how to use sense voice with sherpa-onnx's C++ API
// for streaming speech recognition from a microphone.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
// tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
// rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
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
#include <vector>

#include "portaudio.h"       // NOLINT
#include "sherpa-display.h"  // NOLINT
#include "sherpa-onnx/c-api/cxx-api.h"
#include "sherpa-onnx/csrc/microphone.h"

std::queue<std::vector<float>> samples_queue;
std::condition_variable condition_variable;
std::mutex mutex;
bool stop = false;

static void Handler(int32_t /*sig*/) {
  stop = true;
  condition_variable.notify_one();
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void * /*user_data*/) {
  std::lock_guard<std::mutex> lock(mutex);
  samples_queue.emplace(
      reinterpret_cast<const float *>(input_buffer),
      reinterpret_cast<const float *>(input_buffer) + frames_per_buffer);
  condition_variable.notify_one();

  return stop ? paComplete : paContinue;
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

  config.model_config.sense_voice.model =
      "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx";
  config.model_config.sense_voice.use_itn = false;
  config.model_config.sense_voice.language = "auto";
  config.model_config.tokens =
      "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt";

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

int32_t main() {
  signal(SIGINT, Handler);

  using namespace sherpa_onnx::cxx;  // NOLINT

  auto vad = CreateVad();
  auto recognizer = CreateOfflineRecognizer();

  sherpa_onnx::Microphone mic;

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  if (num_devices == 0) {
    std::cerr << "  If you are using Linux, please try "
                 "./build/bin/sense-voice-simulate-streaming-alsa-cxx-api\n";
    return -1;
  }

  int32_t device_index = Pa_GetDefaultInputDevice();
  const char *pDeviceIndex = std::getenv("SHERPA_ONNX_MIC_DEVICE");
  if (pDeviceIndex) {
    fprintf(stderr, "Use specified device: %s\n", pDeviceIndex);
    device_index = atoi(pDeviceIndex);
  }
  mic.PrintDevices(device_index);

  float mic_sample_rate = 16000;
  const char *sample_rate_str = std::getenv("SHERPA_ONNX_MIC_SAMPLE_RATE");
  if (sample_rate_str) {
    fprintf(stderr, "Use sample rate %f for mic\n", mic_sample_rate);
    mic_sample_rate = atof(sample_rate_str);
  }
  float sample_rate = 16000;
  LinearResampler resampler;
  if (mic_sample_rate != sample_rate) {
    float min_freq = std::min(mic_sample_rate, sample_rate);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler = LinearResampler::Create(mic_sample_rate, sample_rate,
                                        lowpass_cutoff, lowpass_filter_width);
  }
  if (!mic.OpenDevice(device_index, mic_sample_rate, 1, RecordCallback,
                      nullptr)) {
    std::cerr << "Failed to open microphone device\n";
    return -1;
  }

  int32_t window_size = 512;  // samples, please don't change

  int32_t offset = 0;
  std::vector<float> buffer;
  bool speech_started = false;

  auto started_time = std::chrono::steady_clock::now();

  SherpaDisplay display;

  std::cout << "Started! Please speak\n";

  while (!stop) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      while (samples_queue.empty() && !stop) {
        condition_variable.wait(lock);
      }

      const auto &s = samples_queue.front();
      if (!resampler.Get()) {
        buffer.insert(buffer.end(), s.begin(), s.end());
      } else {
        auto resampled = resampler.Resample(s.data(), s.size(), false);
        buffer.insert(buffer.end(), resampled.begin(), resampled.end());
      }

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
      stream.AcceptWaveform(sample_rate, buffer.data(), buffer.size());

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
      stream.AcceptWaveform(sample_rate, segment.samples.data(),
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

  return 0;
}
