// cxx-api-examples/forced-alignment-zh-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates forced alignment of Chinese speech with
// sherpa-onnx's C++ API. It aligns the test wav of the zh-aishell3-pinyin
// model package with its transcript and writes a Praat TextGrid.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
// tar xvf zh-aishell3-pinyin.tar.bz2
// rm zh-aishell3-pinyin.tar.bz2
//
// clang-format on

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  const std::string model_dir = "./zh-aishell3-pinyin";
  const std::string wav_filename = "./zh-aishell3-pinyin/zh.wav";

  // The transcript contains hanzi; the file must be saved as UTF-8.
  const std::string text =
      "不要问你的国家能为你做什么，而要问你能为你的国家做什么。";

  Wave wave = ReadWave(wav_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read " << wav_filename << "\n";
    return -1;
  }

  ForcedAlignerConfig config;
  config.model_dir = model_dir;
  config.lexicon = model_dir + "/lexicon.txt";

  ForcedAligner aligner = ForcedAligner::Create(config);
  OfflineStream stream = aligner.CreateStream();

  // Optional: per-stream options, e.g. debug output of one utterance.
  stream.SetOption("debug", "1");

  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        static_cast<int32_t>(wave.samples.size()));

  ForcedAlignmentResult result = aligner.Compute(stream, text);

  std::cout << "duration: " << result.duration << " seconds\n";
  for (const auto &tier : result.tiers) {
    std::cout << "tier " << tier.name << ": " << tier.intervals.size()
              << " intervals\n";
  }

  const std::string textgrid_filename = "zh.TextGrid";
  std::ofstream os(textgrid_filename);
  os << result.textgrid;
  os.close();
  std::cout << "\n" << textgrid_filename << " created!\n";

  return 0;
}
