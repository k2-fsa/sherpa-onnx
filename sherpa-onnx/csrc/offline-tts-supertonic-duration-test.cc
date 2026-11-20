// sherpa-onnx/csrc/offline-tts-supertonic-duration-test.cc
//
// Copyright (c)  2026  linzch3

#include "sherpa-onnx/csrc/offline-tts-supertonic-duration.h"

#include <cstdint>
#include <limits>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(SupertonicDuration, RejectsNonFiniteOrNonPositiveDuration) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();

  for (float speed : {1.0f, 1.5f}) {
    float dur = inf;
    EXPECT_FALSE(AdjustSupertonicDuration(speed, &dur));

    dur = -inf;
    EXPECT_FALSE(AdjustSupertonicDuration(speed, &dur));

    dur = nan;
    EXPECT_FALSE(AdjustSupertonicDuration(speed, &dur));

    dur = 0.0f;
    EXPECT_FALSE(AdjustSupertonicDuration(speed, &dur));

    dur = -1.0f;
    EXPECT_FALSE(AdjustSupertonicDuration(speed, &dur));
  }
}

TEST(SupertonicDuration, RejectsDurationThatOverflowsAfterSpeed) {
  // 3e38 / 0.5 is larger than FLT_MAX, so it becomes inf.
  float dur = 3e38f;
  EXPECT_FALSE(AdjustSupertonicDuration(0.5f, &dur));
}

TEST(SupertonicDuration, RejectsDurationThatBecomesZeroAfterSpeed) {
  // 1 / inf is 0, which must not be clamped to the minimum duration.
  float dur = 1.0f;
  EXPECT_FALSE(
      AdjustSupertonicDuration(std::numeric_limits<float>::infinity(), &dur));
}

TEST(SupertonicDuration, AppliesSpeedAndMinDuration) {
  float dur = 2.0f;
  EXPECT_TRUE(AdjustSupertonicDuration(1.0f, &dur));
  EXPECT_FLOAT_EQ(dur, 2.0f);

  dur = 2.0f;
  EXPECT_TRUE(AdjustSupertonicDuration(2.0f, &dur));
  EXPECT_FLOAT_EQ(dur, 1.0f);

  // 0.1 / 4 = 0.025 is below the minimum duration of 0.1 s
  dur = 0.1f;
  EXPECT_TRUE(AdjustSupertonicDuration(4.0f, &dur));
  EXPECT_FLOAT_EQ(dur, kSupertonicMinDuration);
}

TEST(SupertonicDuration, WavLengthIsCapped) {
  // Any cap works here; the impl passes kMaxLatentLen * chunk size.
  const int64_t max_wav_len = 30720000;
  const int32_t sample_rate = 44100;

  // 1.5 s at 44100 Hz
  EXPECT_EQ(SupertonicDurationToWavLength(1.5f, sample_rate, max_wav_len),
            66150);

  // Tiny durations still give at least one sample
  EXPECT_EQ(SupertonicDurationToWavLength(1e-9f, sample_rate, max_wav_len), 1);

  // Large but finite durations used to overflow int64_t when converted. They
  // must be capped so that the output buffer can be reserved.
  EXPECT_EQ(SupertonicDurationToWavLength(1e6f, sample_rate, max_wav_len),
            max_wav_len);
  EXPECT_EQ(SupertonicDurationToWavLength(3e38f, sample_rate, max_wav_len),
            max_wav_len);
}

}  // namespace sherpa_onnx
