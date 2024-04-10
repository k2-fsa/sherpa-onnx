// sherpa-onnx/csrc/sherpa-onnx-microphone-offline-audio-tagging.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cctype>  // std::tolower
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "portaudio.h"  // NOLINT
#include "sherpa-onnx/csrc/audio-tagging.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/microphone.h"

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

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void *user_data) {
  std::lock_guard<std::mutex> lock(samples_mutex);

  auto p = reinterpret_cast<const float *>(input_buffer);
  samples.insert(samples.end(), p, p + frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Press Enter to exit\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
Audio tagging from microphone.
Usage:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
tar xvf sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
rm sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2

./bin/sherpa-onnx-microphone-offline-audio-tagging \
  --zipformer-model=./sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.onnx \
  --labels=./sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv

Please see
https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
for more models.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::AudioTaggingConfig config;
  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 0) {
    fprintf(stderr, "\nThis program does not support positional arguments\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  SHERPA_ONNX_LOGE("Creating audio tagger ...");
  sherpa_onnx::AudioTagging tagger(config);
  SHERPA_ONNX_LOGE("Audio tagger created created!");

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
  fprintf(stderr, "Started\n");

  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  std::thread t(DetectKeyPress);
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

        SHERPA_ONNX_LOGE("Computing...");
        auto s = tagger.CreateStream();
        s->AcceptWaveform(sample_rate, buf.data(), buf.size());
        auto results = tagger.Compute(s.get());

        SHERPA_ONNX_LOGE("Result is:");

        int32_t i = 0;
        std::ostringstream os;
        for (const auto &event : results) {
          os << i << ": " << event.ToString() << "\n";
          i += 1;
        }

        SHERPA_ONNX_LOGE("\n%s\n", os.str().c_str());

        state = State::kIdle;
        SHERPA_ONNX_LOGE("Press Enter to start");
        break;
      }
    }

    Pa_Sleep(20);  // sleep for 20ms
  }
  t.join();

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  return 0;
}
