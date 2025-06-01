// sherpa-onnx/csrc/sherpa-onnx-keyword-spotter-microphone.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

#include "portaudio.h"  // NOLINT
#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/microphone.h"

bool stop = false;
float mic_sample_rate = 16000;

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void *user_data) {
  auto stream = reinterpret_cast<sherpa_onnx::OnlineStream *>(user_data);

  stream->AcceptWaveform(mic_sample_rate,
                         reinterpret_cast<const float *>(input_buffer),
                         frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t /*sig*/) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program uses streaming models with microphone for keyword spotting.
Usage:

  ./bin/sherpa-onnx-keyword-spotter-microphone \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --provider=cpu \
    --num-threads=1 \
    --keywords-file=keywords.txt

Please refer to
https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::KeywordSpotterConfig config;

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

  sherpa_onnx::KeywordSpotter spotter(config);
  auto s = spotter.CreateStream();

  sherpa_onnx::Microphone mic;

  int32_t device_index = Pa_GetDefaultInputDevice();
  if (device_index == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    fprintf(stderr, "If you are using Linux, please switch to \n");
    fprintf(stderr, " ./bin/sherpa-onnx-keyword-spotter-alsa \n");
    exit(EXIT_FAILURE);
  }

  const char *pDeviceIndex = std::getenv("SHERPA_ONNX_MIC_DEVICE");
  if (pDeviceIndex) {
    fprintf(stderr, "Use specified device: %s\n", pDeviceIndex);
    device_index = atoi(pDeviceIndex);
  }

  mic.PrintDevices(device_index);

  const char *pSampleRateStr = std::getenv("SHERPA_ONNX_MIC_SAMPLE_RATE");
  if (pSampleRateStr) {
    fprintf(stderr, "Use sample rate %f for mic\n", mic_sample_rate);
    mic_sample_rate = atof(pSampleRateStr);
  }

  if (!mic.OpenDevice(device_index, mic_sample_rate, 1, RecordCallback,
                      s.get())) {
    fprintf(stderr, "portaudio error: %d\n", device_index);
    exit(EXIT_FAILURE);
  }

  int32_t keyword_index = 0;
  sherpa_onnx::Display display;
  while (!stop) {
    while (spotter.IsReady(s.get())) {
      spotter.DecodeStream(s.get());

      const auto r = spotter.GetResult(s.get());
      if (!r.keyword.empty()) {
        display.Print(keyword_index, r.AsJsonString());
        fflush(stderr);
        keyword_index++;

        spotter.Reset(s.get());
      }
    }

    Pa_Sleep(20);  // sleep for 20ms
  }

  return 0;
}
