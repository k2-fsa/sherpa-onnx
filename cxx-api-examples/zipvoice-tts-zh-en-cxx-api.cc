// cxx-api-examples/zipvoice-tts-zh-en-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

// This file shows how to use sherpa-onnx CXX API
// for Chinese/English zero-shot TTS with ZipVoice.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
rm sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

./zipvoice-tts-zh-en-cxx-api
*/
// clang-format on

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

#include "sherpa-onnx/c-api/cxx-api.h"

static int32_t ProgressCallback(const float *samples, int32_t num_samples,
                                float progress, void *arg) {
  fprintf(stderr, "Progress: %.3f%%\n", progress * 100);
  // return 1 to continue generating
  // return 0 to stop generating
  return 1;
}

int32_t main(int32_t argc, char *argv[]) {
  using namespace sherpa_onnx::cxx;  // NOLINT
  OfflineTtsConfig config;

  config.model.zipvoice.encoder =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx";
  config.model.zipvoice.decoder =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx";
  config.model.zipvoice.data_dir =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data";
  config.model.zipvoice.lexicon =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt";
  config.model.zipvoice.tokens =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt";
  config.model.zipvoice.vocoder = "./vocos_24khz.onnx";

  config.model.num_threads = 2;

  // If you want to see debug messages, please set it to 1
  config.model.debug = 0;

  std::string filename = "./generated-zipvoice-zh-en-cxx.wav";
  std::string text =
      "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, "
      "就是全心投入并享受其中.";
  std::string reference_text =
      "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系.";
  std::string reference_audio_file =
      "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav";

  auto tts = OfflineTts::Create(config);

  GenerationConfig gen_config;
  gen_config.speed = 1.0;
  gen_config.num_steps = 4;
  gen_config.reference_text = reference_text;
  gen_config.extra["min_char_in_sentence"] = "10";

  Wave wave = ReadWave(reference_audio_file);
  gen_config.reference_audio = std::move(wave.samples);
  gen_config.reference_sample_rate = wave.sample_rate;

#if 0
  // If you don't want to use a callback, then please enable this branch
  GeneratedAudio audio = tts.Generate(text, gen_config);
#else
  GeneratedAudio audio = tts.Generate(text, gen_config, ProgressCallback);
#endif

  WriteWave(filename, {audio.samples, audio.sample_rate});

  fprintf(stderr, "Input text is: %s\n", text.c_str());
  fprintf(stderr, "Saved to: %s\n", filename.c_str());

  return 0;
}
