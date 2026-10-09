// sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl-test.cc
//
// Copyright (c)  2026  fra-shipper

#include "sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

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

// SplitQwen3AlignerWords mirrors tokenize_space_lang() in the reference
// qwen3_forced_aligner.py: whitespace-delimited segments are stripped of
// non-letter/number characters (keeping "'"), then CJK ideographs are split
// into per-character words.
TEST(SplitQwen3AlignerWords, EnglishWords) {
  auto words = SplitQwen3AlignerWords("Hello, world!  it's fine.");
  ASSERT_EQ(words.size(), 4);
  EXPECT_EQ(words[0], "Hello");
  EXPECT_EQ(words[1], "world");
  EXPECT_EQ(words[2], "it's");
  EXPECT_EQ(words[3], "fine");
}

TEST(SplitQwen3AlignerWords, CjkCharsBecomeWords) {
  auto words = SplitQwen3AlignerWords("你好，世界");
  ASSERT_EQ(words.size(), 4);
  EXPECT_EQ(words[0], "你");
  EXPECT_EQ(words[1], "好");
  EXPECT_EQ(words[2], "世");
  EXPECT_EQ(words[3], "界");
}

TEST(SplitQwen3AlignerWords, MixedChineseLatin) {
  auto words = SplitQwen3AlignerWords("今天 is 2026年9月");
  ASSERT_EQ(words.size(), 7);
  EXPECT_EQ(words[0], "今");
  EXPECT_EQ(words[1], "天");
  EXPECT_EQ(words[2], "is");
  EXPECT_EQ(words[3], "2026");
  EXPECT_EQ(words[4], "年");
  EXPECT_EQ(words[5], "9");
  EXPECT_EQ(words[6], "月");
}

TEST(SplitQwen3AlignerWords, HangulStaysInWord) {
  // Hangul syllables are not Han ideographs; the reference keeps them whole.
  auto words = SplitQwen3AlignerWords("안녕하세요 세계");
  ASSERT_EQ(words.size(), 2);
  EXPECT_EQ(words[0], "안녕하세요");
  EXPECT_EQ(words[1], "세계");
}

TEST(SplitQwen3AlignerWords, EmptyAndPunctOnly) {
  EXPECT_TRUE(SplitQwen3AlignerWords("").empty());
  EXPECT_TRUE(SplitQwen3AlignerWords("   ").empty());
  EXPECT_TRUE(SplitQwen3AlignerWords("!?, .").empty());
}

// FixQwen3AlignerTimestamps ports fix_timestamp() from the reference: it
// finds the longest non-decreasing subsequence and repairs the outliers.
TEST(FixQwen3AlignerTimestamps, AlreadyMonotonic) {
  std::vector<int64_t> data{1, 5, 5, 9, 20};
  FixQwen3AlignerTimestamps(&data);
  EXPECT_EQ(data, (std::vector<int64_t>{1, 5, 5, 9, 20}));
}

TEST(FixQwen3AlignerTimestamps, SingleOutlierSnappedToNeighbor) {
  // 30 breaks monotonicity; a lone outlier snaps to the nearest
  // in-sequence neighbor. Reference fix_timestamp([1,5,30,9,20]) ==
  // [1,5,5,9,20].
  std::vector<int64_t> data{1, 5, 30, 9, 20};
  FixQwen3AlignerTimestamps(&data);
  EXPECT_EQ(data, (std::vector<int64_t>{1, 5, 5, 9, 20}));
}

TEST(FixQwen3AlignerTimestamps, LongAnomalyRunIsInterpolated) {
  // Three consecutive outliers get linear interpolation between the
  // surrounding in-sequence values (left=5, right=10, 3 anomalies ->
  // step = (10-5)/4 = 1.25 -> 6, 7, 8 after truncation).
  // Reference fix_timestamp([1,5,100,99,98,10,20]) == [1,5,6,7,8,10,20].
  std::vector<int64_t> data{1, 5, 100, 99, 98, 10, 20};
  FixQwen3AlignerTimestamps(&data);
  EXPECT_EQ(data, (std::vector<int64_t>{1, 5, 6, 7, 8, 10, 20}));
}

TEST(FixQwen3AlignerTimestamps, TrailingOutlierSnappedToLeft) {
  // The LIS keeps {5,30,31,32}; the trailing 20 has no right neighbor and
  // snaps to the left in-sequence value.
  // Reference fix_timestamp([5,30,31,32,20]) == [5,30,31,32,32].
  std::vector<int64_t> data{5, 30, 31, 32, 20};
  FixQwen3AlignerTimestamps(&data);
  EXPECT_EQ(data, (std::vector<int64_t>{5, 30, 31, 32, 32}));
}

TEST(FixQwen3AlignerTimestamps, EmptyAndSingletonAreNoop) {
  std::vector<int64_t> empty;
  FixQwen3AlignerTimestamps(&empty);
  EXPECT_TRUE(empty.empty());

  std::vector<int64_t> one{42};
  FixQwen3AlignerTimestamps(&one);
  EXPECT_EQ(one[0], 42);
}

}  // namespace sherpa_onnx
