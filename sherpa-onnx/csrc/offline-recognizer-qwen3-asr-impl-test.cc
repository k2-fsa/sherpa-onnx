// sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl-test.cc
//
// Copyright (c)  2026  fra-shipper

#include "sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h"

#include <array>
#include <cstring>

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

}  // namespace sherpa_onnx
