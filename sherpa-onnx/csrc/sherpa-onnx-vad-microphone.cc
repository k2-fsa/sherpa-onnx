// sherpa-onnx/csrc/sherpa-onnx-vad-microphone.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cctype>  // std::tolower
#include <mutex>
#include <queue>

#include "portaudio.h"  // NOLINT
#include "sherpa-onnx/csrc/microphone.h"
#include "sherpa-onnx/csrc/vad-model.h"

bool stop = false;
std::mutex mutex;
std::queue<std::vector<float>> queue;

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void *user_data) {
  int32_t window_size = *reinterpret_cast<int32_t *>(user_data);

  std::lock_guard<std::mutex> lock(mutex);

  std::vector<float> samples(
      reinterpret_cast<const float *>(input_buffer),
      reinterpret_cast<const float *>(input_buffer) + frames_per_buffer);

  if (!queue.empty() && queue.back().size() < window_size) {
    queue.back().insert(queue.back().end(), samples.begin(), samples.end());
  } else {
    queue.push(std::move(samples));
  }

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
    --provider=cpu \
    --num-threads=1

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
                    RecordCallback, &config.silero_vad.window_size);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  err = Pa_StartStream(stream);

  auto vad_model = sherpa_onnx::VadModel::Create(config);

  fprintf(stderr, "Started\n");

  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  int32_t speech_count = 0;
  int32_t non_speech_count = 0;
  while (!stop) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      while (!queue.empty() &&
             queue.front().size() >= config.silero_vad.window_size) {
        bool is_speech =
            vad_model->IsSpeech(queue.front().data(), queue.front().size());

        queue.pop();

        if (is_speech) {
          speech_count += 1;
          non_speech_count = 0;
        } else {
          speech_count = 0;
          non_speech_count += 1;
        }

        if (speech_count == 1) {
          static int32_t k = 0;
          ++k;
          fprintf(stderr, "Detected speech: %d\n", k);
        }

        if (non_speech_count == 1) {
          static int32_t k = 0;
          ++k;
          fprintf(stderr, "Detected non-speech: %d\n", k);
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
