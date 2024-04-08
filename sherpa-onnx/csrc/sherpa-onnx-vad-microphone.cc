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
                              void *user_data) {
  std::lock_guard<std::mutex> lock(mutex);
  buffer.Push(reinterpret_cast<const float *>(input_buffer), frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t sig) {
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
https://github.com/snakers4/silero-vad/blob/master/files/silero_vad.onnx

For instance, use
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
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

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "Num devices: %d\n", num_devices);

  PaStreamParameters param;

  param.device = Pa_GetDefaultInputDevice();
  if (param.device == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Use default device: %d\n", param.device);

  const PaDeviceInfo *info = Pa_GetDeviceInfo(param.device);
  fprintf(stderr, "  Name: %s\n", info->name);
  fprintf(stderr, "  Max input channels: %d\n", info->maxInputChannels);

  param.channelCount = 1;
  param.sampleFormat = paFloat32;

  param.suggestedLatency = info->defaultLowInputLatency;
  param.hostApiSpecificStreamInfo = nullptr;
  float sample_rate = 16000;

  PaStream *stream;
  PaError err =
      Pa_OpenStream(&stream, &param, nullptr, /* &outputParameters, */
                    sample_rate,
                    0,          // frames per buffer
                    paClipOff,  // we won't output out of range samples
                                // so don't bother clipping them
                    RecordCallback, nullptr);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  err = Pa_StartStream(stream);

  auto vad = std::make_unique<sherpa_onnx::VoiceActivityDetector>(config);

  fprintf(stderr, "Started\n");

  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  int32_t window_size = config.silero_vad.window_size;
  bool printed = false;

  int32_t k = 0;
  while (!stop) {
    {
      std::lock_guard<std::mutex> lock(mutex);

      while (buffer.Size() >= window_size) {
        std::vector<float> samples = buffer.Get(buffer.Head(), window_size);
        buffer.Pop(window_size);
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
          sherpa_onnx::WriteWave(filename, 16000, segment.samples.data(),
                                 segment.samples.size());
          fprintf(stderr, "Saved to %s\n", filename);
          fprintf(stderr, "----------\n");

          vad->Pop();
        }
      }
    }
    Pa_Sleep(100);  // sleep for 100ms
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  return 0;
}
