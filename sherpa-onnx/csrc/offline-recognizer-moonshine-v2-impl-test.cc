// sherpa-onnx/csrc/offline-recognizer-moonshine-v2-impl-test.cc
//
// Copyright (c)  2026  kyo-zzz

#include "sherpa-onnx/csrc/offline-recognizer-moonshine-v2-impl.h"

#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

// DecodeStreams() returns an empty transcript for digitally silent audio so
// that the decoder cannot hallucinate text (e.g. "You") for it. These tests
// cover the raw-sample silence detector.
TEST(MoonshineV2AudioIsSilent, ZerosAreSilence) {
  std::vector<float> samples(16000, 0.0f);  // 1 s of digital silence
  EXPECT_TRUE(MoonshineV2AudioIsSilent(samples.data(), samples.size()));
}

TEST(MoonshineV2AudioIsSilent, AnyNonZeroSampleIsSignal) {
  std::vector<float> samples(16000, 0.0f);
  samples[8000] = 0.25f;
  EXPECT_FALSE(MoonshineV2AudioIsSilent(samples.data(), samples.size()));
}

TEST(MoonshineV2AudioIsSilent, FloatNoiseAroundZeroIsSilence) {
  std::vector<float> samples = {1e-9f, -1e-9f, 0.0f, 5e-8f};
  EXPECT_TRUE(MoonshineV2AudioIsSilent(samples.data(), samples.size()));
}

TEST(MoonshineV2AudioIsSilent, SingleSample) {
  std::vector<float> zero = {0.0f};
  EXPECT_TRUE(MoonshineV2AudioIsSilent(zero.data(), zero.size()));

  std::vector<float> signal = {0.5f};
  EXPECT_FALSE(MoonshineV2AudioIsSilent(signal.data(), signal.size()));
}

TEST(MoonshineV2AudioIsSilent, EmptyInputIsNotSilence) {
  // An empty clip is handled by the caller, not here.
  float unused = 0;
  EXPECT_FALSE(MoonshineV2AudioIsSilent(&unused, 0));
}

}  // namespace sherpa_onnx
