// sherpa-onnx/csrc/offline-recognizer-cohere-transcribe-impl-test.cc
//
// Copyright (c)  2026  kyo-zzz

#include "sherpa-onnx/csrc/offline-recognizer-cohere-transcribe-impl.h"

#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

// DecodeStreams() returns an empty transcript for digitally silent audio so
// that the decoder cannot hallucinate sentences for it (measured: 7 of 8
// supported languages hallucinate, e.g. "Herr Präsident, meine Damen und
// Herren!" for de). These tests cover the silence detector.
TEST(CohereHasSignal, SilenceVsSignal) {
  float unused = 0;
  EXPECT_FALSE(CohereHasSignal(&unused, 0));

  // digital silence: normalized features stay far below the threshold
  std::vector<float> silence = {0.0f, 0.05f, -0.087f, 0.02f};
  EXPECT_FALSE(CohereHasSignal(silence.data(), silence.size()));

  // any audio with content is well above it
  std::vector<float> signal = {0.0f, 0.05f, -1.5f, 0.02f};
  EXPECT_TRUE(CohereHasSignal(signal.data(), signal.size()));
}

TEST(CohereHasSignal, EmptyFeaturesAreNotSignal) {
  float unused = 0;
  EXPECT_FALSE(CohereHasSignal(&unused, 0));
}

TEST(CohereHasSignal, ThresholdIsStrict) {
  // the comparison is strictly greater-than, so a value equal to
  // kCohereSilenceFeatureAbsMax still counts as silent
  std::vector<float> below = {0.999f};
  EXPECT_FALSE(CohereHasSignal(below.data(), below.size()));

  std::vector<float> exact = {1.0f};
  EXPECT_FALSE(CohereHasSignal(exact.data(), exact.size()));

  std::vector<float> above = {1.001f};
  EXPECT_TRUE(CohereHasSignal(above.data(), above.size()));
}

TEST(CohereHasSignal, LoudSingleSampleIsSignal) {
  std::vector<float> features(1024, 0.001f);
  features[512] = 3.0f;
  EXPECT_TRUE(CohereHasSignal(features.data(), features.size()));
}

}  // namespace sherpa_onnx
