// sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl-test.cc
//
// Copyright (c)  2026  fra-shipper

#include "sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

struct PrefillLogitsCase {
  std::vector<int64_t> shape;
  int32_t context_len;
  int32_t expected;
};

class Qwen3ASRPrefillLogitsIndexTest
    : public ::testing::TestWithParam<PrefillLogitsCase> {};

TEST_P(Qwen3ASRPrefillLogitsIndexTest, SelectsOrRejectsShape) {
  const auto &test = GetParam();
  EXPECT_EQ(GetQwen3ASRPrefillLogitsIndex(test.shape, test.context_len),
            test.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Layouts, Qwen3ASRPrefillLogitsIndexTest,
    ::testing::ValuesIn(std::vector<PrefillLogitsCase>{
        {{1, 8, 7}, 8, 7},
        {{1, 10, 7}, 8, 7},
        {{1, 1, 7}, 8, 0},
        {{1, 1, 7}, 1, 0},
        {{1, 4, 7}, 8, -1},
        {{1, 0, 7}, 8, -1},
        {{1, 1, 0}, 8, -1},
        {{}, 8, -1},
        {{1}, 8, -1},
        {{1, 7}, 8, -1},
        {{1, 1, 7, 1}, 8, -1},
        {{0, 8, 7}, 8, -1},
        {{2, 8, 7}, 8, -1},
        {{-1, 8, 7}, 8, -1},
        {{1, -1, 7}, 8, -1},
        {{1, 8, -1}, 8, -1},
        {{1, int64_t{std::numeric_limits<int32_t>::max()} + 1, 7}, 8, -1},
        {{1, 8, int64_t{std::numeric_limits<int32_t>::max()} + 1}, 8, -1},
        {{1, std::numeric_limits<int64_t>::max(), 7}, 8, -1},
        {{1, 8, std::numeric_limits<int64_t>::max()}, 8, -1},
        {{1, 1, 7}, 0, -1},
        {{1, 1, 7}, -1, -1},
        {{1, std::numeric_limits<int32_t>::max(), 7}, 8, 7},
        {{1, 1, std::numeric_limits<int32_t>::max()}, 8, 0},
    }));

TEST(Qwen3ASRPrefillLogits, SelectedRowHasExpectedTokenScores) {
  constexpr int32_t kVocab = 7;
  constexpr int32_t kEos = 6;
  // Context ends at row 1. Row 2 must not be sampled even when T > context.
  const std::vector<float> full{20, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 8, 0, 0, 9,
                                0, 30, 0, 0, 0, 0, 0};
  const std::vector<float> last{0, 0, 0, 8, 0, 0, 9};
  for (const auto &logits : {full, last}) {
    const std::vector<int64_t> shape{
        1, static_cast<int64_t>(logits.size() / kVocab), kVocab};
    const int32_t index = GetQwen3ASRPrefillLogitsIndex(shape, 2);
    ASSERT_GE(index, 0);
    const auto begin = logits.begin() + static_cast<size_t>(index) * kVocab;
    EXPECT_EQ(std::max_element(begin, begin + kVocab) - begin, kEos);
    // This verifies row access, not the recognizer's private EOS policy.
    std::vector<float> fallback(begin, begin + kVocab);
    fallback[kEos] = -std::numeric_limits<float>::infinity();
    EXPECT_EQ(std::max_element(fallback.begin(), fallback.end()) -
                  fallback.begin(),
              3);
  }
}

// Regression test for https://github.com/k2-fsa/sherpa-onnx/issues/3509
//
// When every frame of audio_features is silence, TrimAudioFeatures() must
// report that via |all_silent| so that GenerateText() can short-circuit to
// an empty result before any hotwords/language prompt tokens are built.
// Previously the all-silent case was indistinguishable from "nothing needed
// trimming", so decoding proceeded and the hotwords/language prompt could
// bias the LLM decoder into hallucinating text for silent audio.
TEST(TrimAudioFeatures, AllSilentSetsFlag) {
  Ort::AllocatorWithDefaultOptions allocator;

  constexpr int32_t kFrames = 5;
  constexpr int32_t kDim = 4;
  std::array<int64_t, 3> shape{1, kFrames, kDim};
  Ort::Value audio_features =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  float *p = audio_features.GetTensorMutableData<float>();
  std::memset(p, 0, sizeof(float) * kFrames * kDim);

  bool all_silent = false;
  Ort::Value trimmed =
      TrimAudioFeatures(std::move(audio_features), allocator, &all_silent);

  EXPECT_TRUE(all_silent);

  auto trimmed_shape = trimmed.GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(trimmed_shape.size(), 3u);
  EXPECT_EQ(trimmed_shape[1], kFrames);
}

TEST(TrimAudioFeatures, TrailingSilenceIsTrimmedAndFlagStaysFalse) {
  Ort::AllocatorWithDefaultOptions allocator;

  constexpr int32_t kFrames = 5;
  constexpr int32_t kValidFrames = 3;
  constexpr int32_t kDim = 4;
  std::array<int64_t, 3> shape{1, kFrames, kDim};
  Ort::Value audio_features =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  float *p = audio_features.GetTensorMutableData<float>();
  std::memset(p, 0, sizeof(float) * kFrames * kDim);
  for (int32_t a = 0; a < kValidFrames; ++a) {
    p[a * kDim] = 1.0f;
  }

  bool all_silent = false;
  Ort::Value trimmed =
      TrimAudioFeatures(std::move(audio_features), allocator, &all_silent);

  EXPECT_FALSE(all_silent);

  auto trimmed_shape = trimmed.GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(trimmed_shape.size(), 3u);
  EXPECT_EQ(trimmed_shape[1], kValidFrames);
}

TEST(TrimAudioFeatures, NoTrailingSilenceFlagStaysFalse) {
  Ort::AllocatorWithDefaultOptions allocator;

  constexpr int32_t kFrames = 3;
  constexpr int32_t kDim = 4;
  std::array<int64_t, 3> shape{1, kFrames, kDim};
  Ort::Value audio_features =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  float *p = audio_features.GetTensorMutableData<float>();
  for (int32_t i = 0; i < kFrames * kDim; ++i) {
    p[i] = 1.0f;
  }

  bool all_silent = false;
  Ort::Value trimmed =
      TrimAudioFeatures(std::move(audio_features), allocator, &all_silent);

  EXPECT_FALSE(all_silent);

  auto trimmed_shape = trimmed.GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(trimmed_shape.size(), 3u);
  EXPECT_EQ(trimmed_shape[1], kFrames);
}

}  // namespace sherpa_onnx
