// cxx-api-examples/vad-cxx-api.cc
//
// Copyright (c)  2025  Xiaomi Corporation

//
// This file demonstrates how to use VAD to remove silences from a file
// clang-format off
//
// To use silero-vad:
//  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
//
// To use ten-vad:
//  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav
//
// clang-format on
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  std::string wave_filename = "./lei-jun-test.wav";
  if (!FileExists(wave_filename)) {
    fprintf(stderr, "Please download %s\n", wave_filename.c_str());
    return -1;
  }

  std::string vad_filename;
  bool use_silero_vad = false;
  bool use_ten_vad = false;

  if (FileExists("./silero_vad.onnx")) {
    printf("Use silero-vad\n");
    vad_filename = "./silero_vad.onnx";
    use_silero_vad = true;
  } else if (FileExists("./ten-vad.onnx")) {
    printf("Use ten-vad\n");
    vad_filename = "./ten-vad.onnx";
    use_ten_vad = true;
  } else {
    fprintf(stderr, "Please provide either silero_vad.onnx or ten-vad.onnx\n");
    return -1;
  }

  VadModelConfig config;
  if (use_silero_vad) {
    config.silero_vad.model = vad_filename;
    config.silero_vad.threshold = 0.3;
    config.silero_vad.min_silence_duration = 0.5;
    config.silero_vad.min_speech_duration = 0.25;
    config.silero_vad.max_speech_duration = 20;
    config.silero_vad.window_size = 512;
  } else if (use_ten_vad) {
    config.ten_vad.model = vad_filename;
    config.ten_vad.threshold = 0.3;
    config.ten_vad.min_silence_duration = 0.5;
    config.ten_vad.min_speech_duration = 0.25;
    config.ten_vad.max_speech_duration = 20;
    config.ten_vad.window_size = 256;
  }

  config.sample_rate = 16000;
  config.debug = true;

  VoiceActivityDetector vad = VoiceActivityDetector::Create(config, 20);
  if (!vad.Get()) {
    std::cerr << "Failed to create VAD. Please check your config\n";
    return -1;
  }

  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }
  bool is_eof = false;
  int32_t i = 0;
  int32_t window_size = use_silero_vad ? config.silero_vad.window_size
                                       : config.ten_vad.window_size;

  int32_t sample_rate = config.sample_rate;

  std::vector<float> samples_without_silence;

  while (!is_eof) {
    if (i + window_size < wave.samples.size()) {
      vad.AcceptWaveform(wave.samples.data() + i, window_size);
      i += window_size;
    } else {
      is_eof = true;
      vad.Flush();
    }

    while (!vad.IsEmpty()) {
      auto segment = vad.Front();
      float start_time = segment.start / static_cast<float>(sample_rate);
      float end_time =
          start_time + segment.samples.size() / static_cast<float>(sample_rate);
      printf("%.3f -- %.3f\n", start_time, end_time);

      samples_without_silence.insert(samples_without_silence.end(),
                                     segment.samples.begin(),
                                     segment.samples.end());

      vad.Pop();
    }
  }

  bool ok = WriteWave("./lei-jun-test-no-silence.wav",
                      {samples_without_silence, sample_rate});
  if (ok) {
    std::cout << "Saved to ./lei-jun-test-no-silence.wav\n";
  } else {
    std::cerr << "Failed to write ./lei-jun-test-no-silence.wav\n";
  }

  return 0;
}
