// sherpa-onnx/csrc/sherpa-onnx-vad-microphone-offline-asr.cc
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
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"

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
This program shows how to use a streaming VAD with non-streaming ASR in
sherpa-onnx.

Please download silero_vad.onnx from
https://github.com/snakers4/silero-vad/blob/master/files/silero_vad.onnx

For instance, use
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx

Please refer to ./sherpa-onnx-microphone-offline.cc
to download models for offline ASR.

(1) Transducer from icefall

  ./bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx

(2) Paraformer from FunASR

  ./bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    --num-threads=1

(3) Whisper models

  ./bin/sherpa-onnx-vad-microphone-offline-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --whisper-encoder=./sherpa-onnx-whisper-base.en/base.en-encoder.int8.onnx \
    --whisper-decoder=./sherpa-onnx-whisper-base.en/base.en-decoder.int8.onnx \
    --tokens=./sherpa-onnx-whisper-base.en/base.en-tokens.txt \
    --num-threads=1
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::VadModelConfig vad_config;

  sherpa_onnx::OfflineRecognizerConfig asr_config;

  vad_config.Register(&po);
  asr_config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 0) {
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
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  auto vad = std::make_unique<sherpa_onnx::VoiceActivityDetector>(vad_config);

  fprintf(stderr, "Started. Please speak\n");

  int32_t window_size = vad_config.silero_vad.window_size;
  int32_t index = 0;

  while (!stop) {
    {
      std::lock_guard<std::mutex> lock(mutex);

      while (buffer.Size() >= window_size) {
        std::vector<float> samples = buffer.Get(buffer.Head(), window_size);
        buffer.Pop(window_size);
        vad->AcceptWaveform(samples.data(), samples.size());
      }
    }

    while (!vad->Empty()) {
      auto &segment = vad->Front();
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

    Pa_Sleep(100);  // sleep for 100ms
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  return 0;
}
