// sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl-test.cc
//
// Copyright (c)  2026  fra-shipper

#include "sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h"

#include <array>
#include <cmath>
#include <cstring>
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

TEST(Qwen3ASRWaveform, RetentionDoesNotChangeFeatures) {
  std::vector<float> samples(16000);
  for (size_t i = 0; i != samples.size(); ++i) {
    samples[i] = 0.1f * std::sin(i * 0.1f);
  }
  WhisperTag tag;
  tag.dim = 128;
  tag.align_to_stft_center = true;
  OfflineStream original(tag);
  tag.keep_waveform = true;
  OfflineStream retained(tag);
  original.AcceptWaveform(16000, samples.data(), samples.size());
  retained.AcceptWaveform(16000, samples.data(), samples.size());
  EXPECT_TRUE(original.GetWaveform().empty());
  EXPECT_EQ(retained.GetWaveform(), samples);
  EXPECT_EQ(original.GetFrames(), retained.GetFrames());
}

TEST(Qwen3ASRWaveform, RetainsResampledAudioWithoutCenterPadding) {
  std::vector<float> samples(8000, 0.1f);
  WhisperTag tag;
  tag.dim = 128;
  tag.align_to_stft_center = true;
  tag.keep_waveform = true;
  OfflineStream retained(tag);
  retained.AcceptWaveform(8000, samples.data(), samples.size());
  const auto &waveform = retained.GetWaveform();
  ASSERT_EQ(waveform.size(), 16000u);
  EXPECT_NEAR(waveform[waveform.size() / 2], 0.1f, 1e-5f);
}

}  // namespace sherpa_onnx
