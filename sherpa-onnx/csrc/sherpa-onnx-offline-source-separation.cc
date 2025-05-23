// sherpa-onnx/csrc/sherpa-onnx-offline-source-separation.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include <stdio.h>

#include <chrono>  // NOLINT
#include <string>

#include "sherpa-onnx/csrc/offline-source-separation.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Non-streaming source separation with sherpa-onnx.

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models
to download models.

Usage:

(1) Use spleeter models

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2
tar xvf sherpa-onnx-spleeter-2stems-fp16.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/audio_example.wav

./bin/sherpa-onnx-offline-source-separation \
  --spleeter-vocals=sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx \
  --spleeter-accompaniment=sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx \
  --input-wav=audio_example.wav \
  --output-vocals-wav=output_vocals.wav \
  --output-accompaniment-wav=output_accompaniment.wav
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OfflineSourceSeparationConfig config;

  std::string input_wave;
  std::string output_vocals_wave;
  std::string output_accompaniment_wave;

  config.Register(&po);
  po.Register("input-wav", &input_wave, "Path to input wav.");
  po.Register("output-vocals-wav", &output_vocals_wave,
              "Path to output vocals wav");
  po.Register("output-accompaniment-wav", &output_accompaniment_wave,
              "Path to output accompaniment wav");

  po.Read(argc, argv);
  if (po.NumArgs() != 0) {
    fprintf(stderr, "Please don't give positional arguments\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (input_wave.empty()) {
    fprintf(stderr, "Please provide --input-wav\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (output_vocals_wave.empty()) {
    fprintf(stderr, "Please provide --output-vocals-wav\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (output_accompaniment_wave.empty()) {
    fprintf(stderr, "Please provide --output-accompaniment-wav\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    exit(EXIT_FAILURE);
  }

  bool is_ok = false;
  sherpa_onnx::OfflineSourceSeparationInput input;
  input.samples.data =
      sherpa_onnx::ReadWaveMultiChannel(input_wave, &input.sample_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "Failed to read '%s'\n", input_wave.c_str());
    return -1;
  }

  fprintf(stderr, "Started\n");

  fprintf(stderr, "Input channels: %d\n",
          static_cast<int32_t>(input.samples.data.size()));
  fprintf(stderr, "Input sample rate: %d\n", input.sample_rate);
  fprintf(stderr, "Input sample size: %d\n", (int)input.samples.data[0].size());

  sherpa_onnx::OfflineSourceSeparation sp(config);

  auto output = sp.Process(input);

  is_ok = sherpa_onnx::WriteWave(
      output_vocals_wave, output.sample_rate, output.stems[0].data[0].data(),
      output.stems[0].data[1].data(), output.stems[0].data[0].size());

  if (!is_ok) {
    fprintf(stderr, "Failed to write to '%s'\n", output_vocals_wave.c_str());
    exit(EXIT_FAILURE);
  }

  is_ok = sherpa_onnx::WriteWave(output_accompaniment_wave, output.sample_rate,
                                 output.stems[1].data[0].data(),
                                 output.stems[1].data[1].data(),
                                 output.stems[1].data[0].size());

  if (!is_ok) {
    fprintf(stderr, "Failed to write to '%s'\n",
            output_accompaniment_wave.c_str());
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "Saved to write to '%s' and '%s'\n",
          output_vocals_wave.c_str(), output_accompaniment_wave.c_str());

  return 0;
}
