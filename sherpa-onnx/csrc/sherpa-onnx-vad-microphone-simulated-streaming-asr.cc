// sherpa-onnx/csrc/sherpa-onnx-vad-microphone-simulated-streaming-asr.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "portaudio.h"  // NOLINT
#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/microphone.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/sherpa-display.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"
#include "sherpa-onnx/csrc/wave-writer.h"

std::queue<std::vector<float>> samples_queue;
std::condition_variable condition_variable;
std::mutex mutex;
bool stop = false;

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

static void Handler(int32_t /*sig*/) {
  stop = true;
  condition_variable.notify_one();
  fprintf(stdout, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program shows how to use a streaming VAD with non-streaming ASR in
sherpa-onnx for real-time speech recognition.

(1) SenseVoice

cd /path/to/sherpa-onnx/build

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

./bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
  --silero-vad-model=./silero_vad.onnx \
  --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx \
  --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt

(2) Parakeet TDT 0.6b v2

cd /path/to/sherpa-onnx/build

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
tar xvf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

./bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
  --silero-vad-model=./silero_vad.onnx \
  --encoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx \
  --decoder=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx \
  --joiner=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx \
  --tokens=./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt

(3) Please refer to our doc for more non-streaming ASR models,
e.g., zipformer, paraformer, whisper, etc.

Please first use ./bin/sherpa-onnx-offline to test the RTF of the model.
A model with RTF < 0.2 should work with this program.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::VadModelConfig vad_config;

  sherpa_onnx::OfflineRecognizerConfig asr_config;

  vad_config.Register(&po);
  asr_config.Register(&po);

  int32_t user_device_index = -1;  // -1 means to use default value
  int32_t user_sample_rate = -1;   // -1 means to use default value

  po.Register("mic-device-index", &user_device_index,
              "If provided, we use it to replace the default device index."
              "You can use sherpa-onnx-pa-devs to list available devices");

  po.Register("mic-sample-rate", &user_sample_rate,
              "If provided, we use it to replace the default sample rate."
              "You can use sherpa-onnx-pa-devs to list sample rate of "
              "available devices");

  if (argc == 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  po.Read(argc, argv);
  if (po.NumArgs() != 0) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stdout, "%s\n", vad_config.ToString().c_str());
  fprintf(stdout, "%s\n", asr_config.ToString().c_str());

  if (!vad_config.Validate()) {
    fprintf(stdout, "Errors in vad_config!\n");
    return -1;
  }

  if (!asr_config.Validate()) {
    fprintf(stdout, "Errors in asr_config!\n");
    return -1;
  }

  fprintf(stdout, "Creating recognizer ...\n");
  sherpa_onnx::OfflineRecognizer recognizer(asr_config);
  fprintf(stdout, "Recognizer created!\n");

  sherpa_onnx::Microphone mic;

  int32_t device_index = Pa_GetDefaultInputDevice();
  if (device_index == paNoDevice) {
    fprintf(stdout, "No default input device found\n");
    exit(EXIT_FAILURE);
  }

  if (user_device_index >= 0) {
    fprintf(stdout, "Use specified device: %d\n", user_device_index);
    device_index = user_device_index;
  } else {
    fprintf(stdout, "Use default device: %d\n", device_index);
  }

  mic.PrintDevices(device_index);

  float mic_sample_rate = 16000;
  if (user_sample_rate > 0) {
    fprintf(stdout, "Use sample rate %d for mic\n", user_sample_rate);
    mic_sample_rate = user_sample_rate;
  }

  if (!mic.OpenDevice(device_index, mic_sample_rate, 1, RecordCallback,
                      nullptr)) {
    fprintf(stdout, "Failed to open device %d\n", device_index);
    exit(EXIT_FAILURE);
  }

  float sample_rate = 16000;
  std::unique_ptr<sherpa_onnx::LinearResample> resampler;
  if (mic_sample_rate != sample_rate) {
    float min_freq = std::min(mic_sample_rate, sample_rate);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler = std::make_unique<sherpa_onnx::LinearResample>(
        mic_sample_rate, sample_rate, lowpass_cutoff, lowpass_filter_width);
  }

  auto vad = std::make_unique<sherpa_onnx::VoiceActivityDetector>(vad_config);

  int32_t window_size = vad_config.silero_vad.window_size;

  int32_t offset = 0;
  bool speech_started = false;
  std::vector<float> buffer;

  auto started_time = std::chrono::steady_clock::now();
  sherpa_onnx::SherpaDisplay display;

  fprintf(stdout, "Started. Please speak\n");
  std::vector<float> resampled;

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
      if (!resampler) {
        buffer.insert(buffer.end(), s.begin(), s.end());
      } else {
        resampler->Resample(s.data(), s.size(), false, &resampled);
        buffer.insert(buffer.end(), resampled.begin(), resampled.end());
      }

      samples_queue.pop();
    }

    for (; offset + window_size < buffer.size(); offset += window_size) {
      vad->AcceptWaveform(buffer.data() + offset, window_size);
      if (!speech_started && vad->IsSpeechDetected()) {
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
      auto s = recognizer.CreateStream();
      s->AcceptWaveform(sample_rate, buffer.data(), buffer.size());
      recognizer.DecodeStream(s.get());
      const auto &result = s->GetResult();
      display.UpdateText(result.text);
      display.Display();

      started_time = std::chrono::steady_clock::now();
    }

    while (!vad->Empty()) {
      // when stopping speak, this while loop is executed

      vad->Pop();

      display.FinalizeCurrentSentence();
      display.Display();

      buffer.clear();
      offset = 0;
      speech_started = false;
    }
  }

  return 0;
}
