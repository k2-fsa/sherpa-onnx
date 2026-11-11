// sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl-test.cc
//
// Copyright (c)  2026  kyo-zzz

#include "sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

namespace {

std::string ContextPrefix() {
  return
      "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会"
      "留空。\n\n\n"
      "**上下文信息：**\n\n\n";
}

}  // namespace

TEST(FunASRNanoBuildUserPrompt, DefaultsPreserveUserPrompt) {
  OfflineFunASRNanoModelConfig config;
  OfflineStream stream{FeatureExtractorConfig{}};
  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream), "语音转写：");

  config.user_prompt = "前文：乙酸乙酯。\n语音转写：";
  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream), config.user_prompt);
}

TEST(FunASRNanoBuildUserPrompt, MissingOptionsUseConfig) {
  OfflineFunASRNanoModelConfig config;
  config.hotwords = "Sherpa, FunASR";
  config.language = "中文";
  config.itn = false;
  config.user_prompt = "custom prompt";
  OfflineStream stream{FeatureExtractorConfig{}};

  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream),
            ContextPrefix() +
                "热词列表：[Sherpa, FunASR]\n语音转写成中文，不进行文本规整：");
}

TEST(FunASRNanoBuildUserPrompt, HotwordsOverrideAndUseExistingParsing) {
  OfflineFunASRNanoModelConfig config;
  config.hotwords = "default hotword";
  config.language = "中文";
  config.itn = false;
  OfflineStream stream{FeatureExtractorConfig{}};
  stream.SetOption("hotwords", " 酯，Sherpa； FunASR;\n\t");

  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream),
            ContextPrefix() +
                "热词列表：[酯, Sherpa, FunASR]\n"
                "语音转写成中文，不进行文本规整：");
}

TEST(FunASRNanoBuildUserPrompt, LanguageAndItnOverrideIndependently) {
  OfflineFunASRNanoModelConfig config;
  config.language = "中文";
  config.itn = false;
  OfflineStream stream{FeatureExtractorConfig{}};

  stream.SetOption("language", "英文");
  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream),
            "语音转写成英文，不进行文本规整：");
  for (const auto &value : {"1", "2", "-1"}) {
    stream.SetOption("itn", value);
    EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream), "语音转写成英文：");
  }
  stream.SetOption("itn", "0");
  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream),
            "语音转写成英文，不进行文本规整：");
}

TEST(FunASRNanoBuildUserPrompt, UserPromptOverrideIsVerbatim) {
  OfflineFunASRNanoModelConfig config;
  OfflineStream stream{FeatureExtractorConfig{}};
  const std::string prompt = "前文：乙酸乙酯。\n\n语音转写：";
  stream.SetOption("user_prompt", prompt);

  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream), prompt);
}

TEST(FunASRNanoBuildUserPrompt, EmptyOptionsClearConfig) {
  OfflineFunASRNanoModelConfig config;
  config.hotwords = "config hotword";
  config.language = "中文";
  config.itn = false;
  config.user_prompt = "config prompt";
  OfflineStream stream{FeatureExtractorConfig{}};
  stream.SetOption("hotwords", "");
  stream.SetOption("language", "");
  stream.SetOption("itn", "1");

  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream), config.user_prompt);
  stream.SetOption("user_prompt", "stream context");
  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream), "stream context");
  stream.SetOption("user_prompt", "");
  // An explicitly empty user prompt uses the existing standard task prompt,
  // rather than falling back to the nonempty recognizer-level prompt.
  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream), "语音转写：");
}

TEST(FunASRNanoBuildUserPrompt, StructuredOptionsRetainPromptPrecedence) {
  OfflineFunASRNanoModelConfig config;
  OfflineStream stream{FeatureExtractorConfig{}};
  stream.SetOption("user_prompt", "custom context");
  stream.SetOption("hotwords", "酯");
  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream),
            ContextPrefix() + "热词列表：[酯]\n语音转写：");

  stream.SetOption("hotwords", "");
  stream.SetOption("itn", "0");
  EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream),
            "语音转写，不进行文本规整：");
}

TEST(FunASRNanoBuildUserPrompt, InvalidItnUsesConfig) {
  OfflineFunASRNanoModelConfig config;
  OfflineStream stream{FeatureExtractorConfig{}};
  for (const auto &value : {"", "invalid", "1x", "2147483648"}) {
    stream.SetOption("itn", value);
    config.itn = false;
    EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream),
              "语音转写，不进行文本规整：");
    config.itn = true;
    EXPECT_EQ(FunASRNanoBuildUserPrompt(config, stream), config.user_prompt);
  }
}

TEST(FunASRNanoBuildUserPrompt, StreamsAndConfigRemainIndependent) {
  OfflineFunASRNanoModelConfig config;
  config.hotwords = "default hotword";
  config.language = "中文";
  config.itn = false;
  config.user_prompt = "default context";
  const std::string original_config = config.ToString();
  OfflineStream first{FeatureExtractorConfig{}};
  OfflineStream second{FeatureExtractorConfig{}};
  OfflineStream defaults{FeatureExtractorConfig{}};

  first.SetOption("hotwords", "");
  first.SetOption("language", "");
  first.SetOption("itn", "1");
  first.SetOption("user_prompt", "first context");
  second.SetOption("hotwords", "second hotword");
  const std::string default_prompt =
      FunASRNanoBuildUserPrompt(config, defaults);
  const std::string second_prompt =
      ContextPrefix() +
      "热词列表：[second hotword]\n语音转写成中文，不进行文本规整：";

  for (int32_t i = 0; i != 2; ++i) {
    EXPECT_EQ(FunASRNanoBuildUserPrompt(config, first), "first context");
    EXPECT_EQ(FunASRNanoBuildUserPrompt(config, second), second_prompt);
    EXPECT_EQ(FunASRNanoBuildUserPrompt(config, defaults), default_prompt);
  }
  EXPECT_EQ(config.ToString(), original_config);
  EXPECT_FALSE(defaults.HasOption("hotwords"));
  EXPECT_FALSE(second.HasOption("user_prompt"));
  EXPECT_EQ(first.GetOption("user_prompt"), "first context");
}

// Digital silence produces constant fbank frames (dither is disabled for this
// model family), so FunASRNanoAudioIsSilent() must report silence for them
// and signal for any varying content. DecodeStreams() uses this to return an
// empty transcript before hotwords/language prompt tokens can bias the LLM
// decoder into hallucinating text for silent audio, mirroring the Qwen3-ASR
// recognizer.
TEST(FunASRNanoAudioIsSilent, ConstantFramesAreSilence) {
  // LFR stacks feature_dim*window floats per output frame; the values are
  // identical for a constant input.
  std::vector<float> features = {2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f};
  EXPECT_TRUE(FunASRNanoAudioIsSilent(features.data(), features.size()));
}

TEST(FunASRNanoAudioIsSilent, RepeatedNonUniformFramesAreSignal) {
  // alternating values still vary frame to frame, so this is signal
  std::vector<float> features = {1.0f, 2.0f, 1.0f, 2.0f};
  EXPECT_FALSE(FunASRNanoAudioIsSilent(features.data(), features.size()));
}

TEST(FunASRNanoAudioIsSilent, VaryingFramesAreSignal) {
  std::vector<float> features = {2.5f, 2.5f, -1.25f, 2.5f, 2.5f, 2.5f};
  EXPECT_FALSE(FunASRNanoAudioIsSilent(features.data(), features.size()));
}

TEST(FunASRNanoAudioIsSilent, ZerosAreSilence) {
  std::vector<float> features(64, 0.0f);
  EXPECT_TRUE(FunASRNanoAudioIsSilent(features.data(), features.size()));
}

TEST(FunASRNanoAudioIsSilent, SingleValueIsSilence) {
  // A single frame carries no variation to inspect.
  std::vector<float> features = {2.5f};
  EXPECT_TRUE(FunASRNanoAudioIsSilent(features.data(), features.size()));
}

TEST(FunASRNanoAudioIsSilent, EmptyInputIsNotSilence) {
  // An empty input is handled by the caller (num_frames <= 0), not here.
  float unused = 0;
  EXPECT_FALSE(FunASRNanoAudioIsSilent(&unused, 0));
}

}  // namespace sherpa_onnx
