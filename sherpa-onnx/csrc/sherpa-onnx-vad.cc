// sherpa-onnx/csrc/sherpa-onnx-vad.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <iomanip>

#include "sherpa-onnx/csrc/voice-activity-detector.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"

int32_t main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
This program shows how to use VAD in sherpa-onnx
to remove silences from a file.

  ./bin/sherpa-onnx-vad \
    --silero-vad-model=/path/to/silero_vad.onnx \
    /path/to/input.wav
    /path/to/output.wav

Please download silero_vad.onnx from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

For instance, use
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

input.wav should be 16kHz.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::VadModelConfig config;

  config.Register(&po);
  po.Read(argc, argv);
  if (po.NumArgs() != 2) {
    fprintf(
        stderr,
        "Please provide only 2 argument2: the input wav and the output wav\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  std::string wav_filename = po.GetArg(1);
  int32_t sampling_rate = -1;

  bool is_ok = false;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);

  if (!is_ok) {
    fprintf(stderr, "Failed to read '%s'\n", wav_filename.c_str());
    return -1;
  }

  if (sampling_rate != 16000) {
    fprintf(stderr, "Support only 16000Hz. Given: %d\n", sampling_rate);
    return -1;
  }

  auto vad = std::make_unique<sherpa_onnx::VoiceActivityDetector>(config);

  int32_t window_size = config.silero_vad.window_size;

  int32_t i = 0;
  bool is_eof = false;

  std::vector<float> samples_without_silence;

  while (!is_eof) {
    if (i + window_size < samples.size()) {
      vad->AcceptWaveform(samples.data() + i, window_size);
      i += window_size;
    } else {
      vad->Flush();
      is_eof = true;
    }

    while (!vad->Empty()) {
      const auto &segment = vad->Front();
      float start_time = segment.start / static_cast<float>(sampling_rate);
      float end_time = start_time + segment.samples.size() /
                                        static_cast<float>(sampling_rate);

      fprintf(stderr, "%.3f -- %.3f\n", start_time, end_time);
      samples_without_silence.insert(samples_without_silence.end(),
                                     segment.samples.begin(),
                                     segment.samples.end());
      vad->Pop();
    }
  }

  sherpa_onnx::WriteWave(po.GetArg(2), sampling_rate,
                         samples_without_silence.data(),
                         samples_without_silence.size());

  fprintf(stderr, "Saved to %s\n", po.GetArg(2).c_str());

  return 0;
}
