// sherpa-onnx/csrc/offline-whisper-lang-probs-test.cc
//
// Copyright (c)  2026  Matias Lin

#include "sherpa-onnx/csrc/offline-whisper-lang-probs.h"

#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

const std::unordered_map<int32_t, std::string> kID2Lang = {
    {0, "ru"}, {1, "en"}, {2, "nl"}
};

TEST(ComputeWhisperLangProbs, SoftmaxIsCorrectAndStable) {
  const std::vector<float> logits = {1000.0f, 999.0f, 998.0f};
  const std::vector<int32_t> ids = {0, 1, 2};
  std::vector<std::pair<std::string, float>> probs;
  ComputeWhisperLangProbs(logits.data(), ids, kID2Lang, &probs);

  ASSERT_EQ(probs.size(), 3u);
  float sum = 0.0f;
  for (const auto &p : probs) {
    EXPECT_FALSE(std::isnan(p.second));
    EXPECT_FALSE(std::isinf(p.second));
    sum += p.second;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-4);
  EXPECT_GT(probs[0].second, probs[1].second);
  EXPECT_GT(probs[1].second, probs[2].second);
}

TEST(ComputeWhisperLangProbs, EmptyIdsClearsProbs) {
  const std::vector<float> logits = {0.0f, 1.0f, 0.0f};
  const std::vector<int32_t> ids = {};
  std::vector<std::pair<std::string, float>> probs = {{"stale", 0.5f}};
  ComputeWhisperLangProbs(logits.data(), ids, kID2Lang, &probs);
  EXPECT_TRUE(probs.empty());
}

TEST(ComputeWhisperLangProbs, DedupsLangIds) {
  const std::vector<float> logits = {5.0f, 1.0f, 1.0f};
  const std::vector<int32_t> ids = {0, 0, 1};
  std::vector<std::pair<std::string, float>> probs;
  ComputeWhisperLangProbs(logits.data(), ids, kID2Lang, &probs);
  ASSERT_EQ(probs.size(), 2u);
  EXPECT_EQ(probs[0].first, "ru");
  EXPECT_EQ(probs[1].first, "en");
  float sum = probs[0].second + probs[1].second;
  EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST(ComputeWhisperLangProbs, NonFiniteLogitsClearsProbs) {
  const float negative_inf = -std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<int32_t> ids = {0, 1, 2};
  std::vector<std::pair<std::string, float>> probs;

  const std::vector<float> inf_logits = {negative_inf, negative_inf,
                                         negative_inf};
  probs = {{"stale", 0.5f}};
  ComputeWhisperLangProbs(inf_logits.data(), ids, kID2Lang, &probs);
  EXPECT_TRUE(probs.empty());

  const std::vector<float> nan_logits = {nan, nan, nan};
  probs = {{"stale", 0.5f}};
  ComputeWhisperLangProbs(nan_logits.data(), ids, kID2Lang, &probs);
  EXPECT_TRUE(probs.empty());

  const std::vector<float> mixed_logits = {negative_inf, nan, negative_inf};
  probs = {{"stale", 0.5f}};
  ComputeWhisperLangProbs(mixed_logits.data(), ids, kID2Lang, &probs);
  EXPECT_TRUE(probs.empty());
}

TEST(ComputeWhisperLangProbs, NullptrGuardsAreNoOp) {
  const std::vector<float> logits = {0.0f, 1.0f, 0.0f};
  const std::vector<int32_t> ids = {0, 1, 2};
  std::vector<std::pair<std::string, float>> probs = {{"stale", 0.5f}};

  ComputeWhisperLangProbs(nullptr, ids, kID2Lang, &probs);
  ASSERT_EQ(probs.size(), 1u);
  EXPECT_EQ(probs[0].first, "stale");

  ComputeWhisperLangProbs(logits.data(), ids, kID2Lang, nullptr);
}

TEST(ApplyTopKLangProbs, EmptyOptionClears) {
  std::vector<std::pair<std::string, float>> probs = {
      {"ru", 0.6f}, {"en", 0.3f}, {"nl", 0.1f}};
  ApplyTopKLangProbs("", &probs);
  EXPECT_TRUE(probs.empty());
}

TEST(ApplyTopKLangProbs, ZeroClears) {
  std::vector<std::pair<std::string, float>> probs = {
      {"ru", 0.6f}, {"en", 0.3f}, {"nl", 0.1f}};
  ApplyTopKLangProbs("0", &probs);
  EXPECT_TRUE(probs.empty());
}

TEST(ApplyTopKLangProbs, MinusOneKeepsAllSorted) {
  std::vector<std::pair<std::string, float>> probs = {
      {"nl", 0.1f}, {"ru", 0.6f}, {"en", 0.3f}};
  ApplyTopKLangProbs("-1", &probs);
  ASSERT_EQ(probs.size(), 3u);
  EXPECT_EQ(probs[0].first, "ru");
  EXPECT_EQ(probs[1].first, "en");
  EXPECT_EQ(probs[2].first, "nl");
}

TEST(ApplyTopKLangProbs, PositiveKeepsTopN) {
  std::vector<std::pair<std::string, float>> probs = {
      {"nl", 0.1f}, {"ru", 0.6f}, {"en", 0.3f}, {"de", 0.05f}};
  ApplyTopKLangProbs("2", &probs);
  ASSERT_EQ(probs.size(), 2u);
  EXPECT_EQ(probs[0].first, "ru");
  EXPECT_EQ(probs[1].first, "en");
}

TEST(ApplyTopKLangProbs, KLargerThanSizeKeepsAll) {
  std::vector<std::pair<std::string, float>> probs = {
      {"nl", 0.1f}, {"ru", 0.6f}, {"en", 0.3f}};
  ApplyTopKLangProbs("10", &probs);
  ASSERT_EQ(probs.size(), 3u);
  EXPECT_EQ(probs[0].first, "ru");
  EXPECT_EQ(probs[1].first, "en");
  EXPECT_EQ(probs[2].first, "nl");
}

TEST(ApplyTopKLangProbs, UnparseableClears) {
  std::vector<std::pair<std::string, float>> probs = {
      {"ru", 0.6f}, {"en", 0.3f}, {"nl", 0.1f}};
  ApplyTopKLangProbs("abc", &probs);
  EXPECT_TRUE(probs.empty());
}

TEST(ApplyTopKLangProbs, NullptrGuardIsNoOp) {
  ApplyTopKLangProbs("5", nullptr);
}

}  // namespace
}  // namespace sherpa_onnx
