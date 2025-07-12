// sherpa-onnx/csrc/sherpa-onnx-vad-microphone.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <mutex>  // NOLINT

#include "portaudio.h"  // NOLINT
#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/microphone.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"
#include "sherpa-onnx/csrc/wave-writer.h"

bool stop = false;
std::mutex mutex;
sherpa_onnx::CircularBuffer buffer(16000 * 60);

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void * /*user_data*/) {
  std::lock_guard<std::mutex> lock(mutex);
  buffer.Push(reinterpret_cast<const float *>(input_buffer), frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t /*sig*/) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program shows how to use VAD in sherpa-onnx.

  ./bin/sherpa-onnx-vad-microphone \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --vad-provider=cpu \
    --vad-num-threads=1

Please download silero_vad.onnx from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

For instance, use
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::VadModelConfig config;

  config.Register(&po);
  po.Read(argc, argv);
  if (po.NumArgs() != 0) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  sherpa_onnx::Microphone mic;

  int32_t device_index = Pa_GetDefaultInputDevice();
  if (device_index == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    fprintf(stderr, "If you are using Linux, please switch to \n");
    fprintf(stderr, " ./bin/sherpa-onnx-vad-alsa \n");
    exit(EXIT_FAILURE);
  }

  const char *pDeviceIndex = std::getenv("SHERPA_ONNX_MIC_DEVICE");
  if (pDeviceIndex) {
    fprintf(stderr, "Use specified device: %s\n", pDeviceIndex);
    device_index = atoi(pDeviceIndex);
  }
  mic.PrintDevices(device_index);

  float mic_sample_rate = 16000;
  const char *pSampleRateStr = std::getenv("SHERPA_ONNX_MIC_SAMPLE_RATE");
  if (pSampleRateStr) {
    fprintf(stderr, "Use sample rate %f for mic\n", mic_sample_rate);
    mic_sample_rate = atof(pSampleRateStr);
  }
  if (!mic.OpenDevice(device_index, mic_sample_rate, 1, RecordCallback,
                      nullptr)) {
    fprintf(stderr, "Failed to open microphone device %d\n", device_index);
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

  auto vad = std::make_unique<sherpa_onnx::VoiceActivityDetector>(config);

  int32_t window_size = config.silero_vad.window_size;
  bool printed = false;

  int32_t k = 0;
  while (!stop) {
    {
      std::lock_guard<std::mutex> lock(mutex);

      while (buffer.Size() >= window_size) {
        std::vector<float> samples = buffer.Get(buffer.Head(), window_size);
        buffer.Pop(window_size);

        if (resampler) {
          std::vector<float> tmp;
          resampler->Resample(samples.data(), samples.size(), true, &tmp);
          samples = std::move(tmp);
        }

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
          float duration = segment.samples.size() / sample_rate;
          fprintf(stderr, "Duration: %.3f seconds\n", duration);

          char filename[128];
          snprintf(filename, sizeof(filename), "seg-%d-%.3fs.wav", k, duration);
          k += 1;
          sherpa_onnx::WriteWave(filename, sample_rate, segment.samples.data(),
                                 segment.samples.size());
          fprintf(stderr, "Saved to %s\n", filename);
          fprintf(stderr, "----------\n");

          vad->Pop();
        }
      }
    }
    Pa_Sleep(100);  // sleep for 100ms
  }

  return 0;
}
