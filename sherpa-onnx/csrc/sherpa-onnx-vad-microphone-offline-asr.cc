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
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"

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

  int32_t device_index = Pa_GetDefaultInputDevice();
  if (device_index == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    fprintf(stderr,
            "  If you are using Linux, please try "
            "./build/bin/sherpa-onnx-vad-alsa-offline-asr\n");
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
    fprintf(stderr, "Failed to open device %d\n", device_index);
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

  fprintf(stderr, "Started. Please speak\n");

  int32_t window_size = vad_config.silero_vad.window_size;
  int32_t index = 0;

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
      }
    }

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

    Pa_Sleep(100);  // sleep for 100ms
  }

  return 0;
}
