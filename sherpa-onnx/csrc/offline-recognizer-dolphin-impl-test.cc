// sherpa-onnx/csrc/offline-recognizer-dolphin-impl-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-recognizer-dolphin-impl.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

SymbolTable MakeTokens() {
  return SymbolTable(
      std::string("<sos> 0\n<eos> 1\n<notimestamp> 2\n<en> 3\n<US> 4\n"
                  "<zh> 5\n<CN> 6\n<blk> 7\n<unk> 8\n"),
      false);
}

TEST(DolphinPrompt, AutomaticLanguageAndRegion) {
  OfflineDolphinModelConfig config;
  std::vector<int64_t> prompt{99};
  EXPECT_TRUE(ResolveDolphinPrompt(config, MakeTokens(), 7, &prompt));
  EXPECT_TRUE(prompt.empty());
}

TEST(DolphinPrompt, LanguageWithOptionalRegion) {
  OfflineDolphinModelConfig config;
  config.language = "en";
  std::vector<int64_t> prompt;
  ASSERT_TRUE(ResolveDolphinPrompt(config, MakeTokens(), 7, &prompt));
  EXPECT_EQ(prompt, (std::vector<int64_t>{3}));
  config.region = "US";
  ASSERT_TRUE(ResolveDolphinPrompt(config, MakeTokens(), 7, &prompt));
  EXPECT_EQ(prompt, (std::vector<int64_t>{3, 4}));
}

TEST(DolphinPrompt, InvalidUpdatePreservesPreviousPrompt) {
  OfflineDolphinModelConfig config;
  std::vector<int64_t> prompt{5, 6};
  config.language = "unknown";
  EXPECT_FALSE(ResolveDolphinPrompt(config, MakeTokens(), 7, &prompt));
  EXPECT_EQ(prompt, (std::vector<int64_t>{5, 6}));
  config.language = "en";
  config.region = "unknown";
  EXPECT_FALSE(ResolveDolphinPrompt(config, MakeTokens(), 7, &prompt));
  EXPECT_EQ(prompt, (std::vector<int64_t>{5, 6}));
  config.language.clear();
  config.region = "US";
  EXPECT_FALSE(ResolveDolphinPrompt(config, MakeTokens(), 7, &prompt));
  EXPECT_EQ(prompt, (std::vector<int64_t>{5, 6}));
}

TEST(DolphinPrompt, RejectsTokenOutsideDecoderVocabulary) {
  OfflineDolphinModelConfig config;
  config.language = "zh";
  std::vector<int64_t> prompt;
  EXPECT_FALSE(ResolveDolphinPrompt(config, MakeTokens(), 5, &prompt));
  EXPECT_TRUE(prompt.empty());
}

TEST(DolphinPrompt, RejectsSwappedCodesAndControlTokens) {
  OfflineDolphinModelConfig config;
  std::vector<int64_t> prompt;
  // control tokens are lowercase like real language codes, so the blocklist
  // (not the charset check) is what must reject them
  for (const auto &language : {"sos", "eos", "notimestamp", "blk", "unk"}) {
    config.language = language;
    EXPECT_FALSE(ResolveDolphinPrompt(config, MakeTokens(), 9, &prompt));
  }
  // codes that normalize fine but are absent from the vocabulary still reject
  config.language = "cn";
  EXPECT_FALSE(ResolveDolphinPrompt(config, MakeTokens(), 9, &prompt));
  config.language = "en";
  config.region = "JP";
  EXPECT_FALSE(ResolveDolphinPrompt(config, MakeTokens(), 9, &prompt));
}

TEST(DolphinPrompt, NormalizesCaseOfLanguageAndRegion) {
  OfflineDolphinModelConfig config;
  std::vector<int64_t> prompt;
  config.language = "ZH";
  ASSERT_TRUE(ResolveDolphinPrompt(config, MakeTokens(), 9, &prompt));
  EXPECT_EQ(prompt, (std::vector<int64_t>{5}));
  config.language = "en";
  config.region = "Cn";
  ASSERT_TRUE(ResolveDolphinPrompt(config, MakeTokens(), 9, &prompt));
  EXPECT_EQ(prompt, (std::vector<int64_t>{3, 6}));
}

TEST(DolphinRecognizer, ConfigUpdatesPreserveSessionsAndValidPrompt) {
  const char *directory = std::getenv("SHERPA_ONNX_DOLPHIN_TEST_MODELS");
  if (!directory) {
    GTEST_SKIP() << "Generate fixtures with scripts/dolphin/test_attention.py";
  }
  const std::string root(directory);
  OfflineRecognizerConfig config;
  config.model_config.tokens = root + "/tokens.txt";
  config.model_config.dolphin.encoder = root + "/encoder.onnx";
  config.model_config.dolphin.decoder = root + "/decoder.onnx";
  config.model_config.num_threads = 1;
  OfflineRecognizer recognizer(config);
  auto decode = [&]() {
    auto stream = recognizer.CreateStream();
    std::vector<float> samples(16000, 0);
    stream->AcceptWaveform(16000, samples.data(), samples.size());
    recognizer.DecodeStream(stream.get());
    return stream->GetResult();
  };
  EXPECT_EQ(decode().text, "HELLO");
  config.model_config.dolphin.language = "zh";
  config.model_config.dolphin.region = "CN";
  config.model_config.dolphin.encoder = "not-reloaded.onnx";
  recognizer.SetConfig(config);
  EXPECT_EQ(recognizer.GetConfig().model_config.dolphin.encoder,
            root + "/encoder.onnx");
  EXPECT_EQ(decode().text, "NIHAO");
  config.model_config.dolphin.language = "unknown";
  recognizer.SetConfig(config);
  EXPECT_EQ(recognizer.GetConfig().model_config.dolphin.language, "zh");
  EXPECT_EQ(decode().text, "NIHAO");
  config.model_config.dolphin.language.clear();
  recognizer.SetConfig(config);
  EXPECT_EQ(decode().text, "NIHAO");
  config.model_config.dolphin.region.clear();
  recognizer.SetConfig(config);
  EXPECT_EQ(decode().text, "HELLO");
}

}  // namespace
}  // namespace sherpa_onnx
