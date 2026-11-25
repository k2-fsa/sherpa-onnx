// sherpa-onnx/csrc/nomo-pvad-model-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/nomo-pvad-model-config.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/vad-model-config.h"
#include "sherpa-onnx/csrc/vad-model.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"

namespace sherpa_onnx {

namespace {

// Path to nomo-pvad.onnx. It is 72 MB, so we do not download it
// automatically. Tests that need it are skipped when it is absent.
const char *kModelPath = "/tmp/sherpa-test-data/nomo-pvad/nomo-pvad.onnx";

std::string TempPath(const std::string &name) {
  std::string dir = testing::TempDir();
  if (!dir.empty() && dir.back() != '/') {
    dir += "/";
  }
  return dir + "sherpa-onnx-nomo-pvad-" + name;
}

std::string WriteFile(const std::string &name, const std::string &content) {
  std::string filename = TempPath(name);
  std::ofstream os(filename);
  os << content;
  return filename;
}

std::string EmbeddingFileContent() {
  std::string s;
  for (int32_t i = 0; i != 192; ++i) {
    s += "0.5 ";
  }
  return s;
}

bool HasModel() {
  if (!FileExists(kModelPath)) {
    std::cerr << "Skipping: " << kModelPath << " does not exist\n";
    return false;
  }
  return true;
}

VadModelConfig GetModelConfig() {
  VadModelConfig config;
  config.nomo_pvad.model = kModelPath;
  config.nomo_pvad.speaker_embedding = std::vector<float>(192, 1.0f);
  config.nomo_pvad.threshold = 0.7f;
  config.sample_rate = 16000;
  config.num_threads = 1;
  config.provider = "cpu";
  return config;
}

}  // namespace

TEST(NomoPvadModelConfig, Validate) {
  std::string model = WriteFile("model.onnx", "dummy");
  std::string embedding = WriteFile("embedding.txt", EmbeddingFileContent());
  std::string missing_model = TempPath("missing.onnx");
  std::string missing_embedding = TempPath("missing.txt");

  {
    // Neither a model nor an embedding is given.
    NomoPvadModelConfig config;
    EXPECT_FALSE(config.Validate());
  }

  {
    // Only the model is given.
    NomoPvadModelConfig config;
    config.model = model;
    EXPECT_FALSE(config.Validate());
  }

  {
    // Only the embedding file is given.
    NomoPvadModelConfig config;
    config.speaker_embedding_file = embedding;
    EXPECT_FALSE(config.Validate());
  }

  {
    // The model file does not exist.
    NomoPvadModelConfig config;
    config.model = missing_model;
    config.speaker_embedding_file = embedding;
    EXPECT_FALSE(config.Validate());
  }

  {
    // The embedding file does not exist.
    NomoPvadModelConfig config;
    config.model = model;
    config.speaker_embedding_file = missing_embedding;
    EXPECT_FALSE(config.Validate());
  }

  {
    // Both files are given and an invalid threshold is used.
    NomoPvadModelConfig config;
    config.model = model;
    config.speaker_embedding_file = embedding;
    EXPECT_TRUE(config.Validate());

    config.threshold = 1.5f;
    EXPECT_FALSE(config.Validate());
  }

  {
    // A non-positive min silence duration.
    NomoPvadModelConfig config;
    config.model = model;
    config.speaker_embedding_file = embedding;
    config.min_silence_duration = 0.0f;
    EXPECT_FALSE(config.Validate());
  }
}

TEST(NomoPvadModelConfig, ToString) {
  NomoPvadModelConfig config;
  config.model = "a.onnx";
  config.speaker_embedding = std::vector<float>(192, 0.5f);
  config.threshold = 0.75f;

  auto s = config.ToString();
  EXPECT_NE(s.find("a.onnx"), std::string::npos) << s;
  EXPECT_NE(s.find("speaker_embedding_size=192"), std::string::npos) << s;
  EXPECT_NE(s.find("threshold=0.75"), std::string::npos) << s;
}

TEST(VadModelConfig, RegisterNomoPvad) {
  VadModelConfig config;
  ParseOptions po("test");
  config.Register(&po);

  const char *argv[] = {
      "test",
      "--print-args=false",
      "--nomo-pvad-model=./nomo-pvad.onnx",
      "--nomo-pvad-speaker-embedding-file=./embedding.txt",
      "--nomo-pvad-threshold=0.6",
      "--nomo-pvad-min-silence-duration=0.4",
      "--nomo-pvad-min-speech-duration=0.2",
      "--nomo-pvad-max-speech-duration=15",
      "--nomo-pvad-neg-threshold=0.45",
      "--vad-sample-rate=16000",
  };
  po.Read(sizeof(argv) / sizeof(argv[0]), argv);

  EXPECT_EQ(config.nomo_pvad.model, "./nomo-pvad.onnx");
  EXPECT_EQ(config.nomo_pvad.speaker_embedding_file, "./embedding.txt");
  EXPECT_FLOAT_EQ(config.nomo_pvad.threshold, 0.6f);
  EXPECT_FLOAT_EQ(config.nomo_pvad.min_silence_duration, 0.4f);
  EXPECT_FLOAT_EQ(config.nomo_pvad.min_speech_duration, 0.2f);
  EXPECT_FLOAT_EQ(config.nomo_pvad.max_speech_duration, 15.0f);
  EXPECT_FLOAT_EQ(config.nomo_pvad.neg_threshold, 0.45f);
  EXPECT_EQ(config.sample_rate, 16000);
}

TEST(VadModelConfig, BackwardCompatibleConstructor) {
  SileroVadModelConfig silero_vad;
  silero_vad.model = "silero_vad.onnx";

  TenVadModelConfig ten_vad;
  ten_vad.model = "ten-vad.onnx";

  // The old 6-argument constructor must still work and must leave
  // nomo_pvad empty.
  VadModelConfig config(silero_vad, ten_vad, 8000, 2, "cuda", true);
  EXPECT_EQ(config.silero_vad.model, "silero_vad.onnx");
  EXPECT_EQ(config.ten_vad.model, "ten-vad.onnx");
  EXPECT_TRUE(config.nomo_pvad.model.empty());
  EXPECT_EQ(config.sample_rate, 8000);
  EXPECT_EQ(config.num_threads, 2);
  EXPECT_EQ(config.provider, "cuda");
  EXPECT_TRUE(config.debug);

  NomoPvadModelConfig nomo_pvad;
  nomo_pvad.model = "nomo-pvad.onnx";

  VadModelConfig config2(silero_vad, ten_vad, nomo_pvad, 8000, 2, "cuda", true);
  EXPECT_EQ(config2.nomo_pvad.model, "nomo-pvad.onnx");
}

TEST(NomoPvadModel, ModelMetaData) {
  if (!HasModel()) {
    return;
  }

  VadModelConfig config = GetModelConfig();
  ASSERT_TRUE(config.Validate());

  auto model = VadModel::Create(config);
  ASSERT_NE(model, nullptr);

  // 160 ms of audio, as documented in the model meta data.
  EXPECT_EQ(model->WindowSize(), 2560);
  EXPECT_EQ(model->WindowShift(), 2560);

  // 0.25 s and 0.5 s at 16 kHz.
  EXPECT_EQ(model->MinSpeechDurationSamples(), 4000);
  EXPECT_EQ(model->MinSilenceDurationSamples(), 8000);
}

TEST(NomoPvadModel, DigitalSilenceHasNoSpeech) {
  if (!HasModel()) {
    return;
  }

  VadModelConfig config = GetModelConfig();
  VoiceActivityDetector vad(config, 60);

  // Three seconds of digital silence must not produce any speech segment.
  std::vector<float> silence(16000 * 3, 0.0f);
  vad.AcceptWaveform(silence.data(), static_cast<int32_t>(silence.size()));
  vad.Flush();

  EXPECT_TRUE(vad.Empty());
}

}  // namespace sherpa_onnx
