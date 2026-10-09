// sherpa-onnx/csrc/offline-stream-parakeet-test.cc
//
// Copyright (c) 2026 Code Myriad

#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {
namespace {
#include "sherpa-onnx/csrc/offline-stream-parakeet-reference.inc"
FeatureExtractorConfig ParakeetConfig() {
  FeatureExtractorConfig config;
  config.feature_dim = 128;
  config.sampling_rate = 16000;
  config.nemo_normalize_type = "per_feature";
  config.parakeet_reference_frontend = true;
  return config;
}

std::vector<float> Signal(int32_t n) {
  std::vector<float> samples(n);
  // Deterministic broadband input, including energy above the old 7600 Hz
  // cutoff. Integer arithmetic avoids platform-dependent random generators.
  uint32_t state = 42;
  for (auto &sample : samples) {
    state = state * 1664525u + 1013904223u;
    sample = (static_cast<int32_t>(state >> 16) - 32768) / 327680.0f;
  }
  return samples;
}
}  // namespace

TEST(ParakeetFrontend, LengthSilenceAndRepeatability) {
  for (int32_t n : {0, 1, 159, 160, 319, 320, 321, 399, 400, 479, 480, 511, 512,
                    15999, 16000, 16001}) {
    for (bool silence : {false, true}) {
      SCOPED_TRACE(n);
      SCOPED_TRACE(silence);
      auto samples = silence ? std::vector<float>(n, 0) : Signal(n);
      OfflineStream stream(ParakeetConfig());
      if (n) stream.AcceptWaveform(16000, samples.data(), n);
      auto frames = stream.GetFrames();
      EXPECT_EQ(frames.size(), n < 320 ? 0 : (n / 160) * 128);
      EXPECT_EQ(frames, stream.GetFrames());
      for (float f : frames) {
        EXPECT_TRUE(std::isfinite(f));
        if (silence) EXPECT_FLOAT_EQ(f, 0);
      }
    }
  }
}

TEST(ParakeetFrontend, NormalizesOnlyValidFrames) {
  auto samples = Signal(16001);
  OfflineStream stream(ParakeetConfig());
  stream.AcceptWaveform(16000, samples.data(), samples.size());
  auto frames = stream.GetFrames();
  ASSERT_EQ(frames.size(), 100 * 128);
  for (int32_t bin = 0; bin < 128; ++bin) {
    double sum = 0, squares = 0;
    for (int32_t frame = 0; frame < 100; ++frame) {
      double f = frames[frame * 128 + bin];
      sum += f;
      squares += f * f;
    }
    EXPECT_NEAR(sum / 100, 0, 1e-6);
    EXPECT_NEAR(squares / 99, 1, 1e-3);
  }
}

TEST(ParakeetFrontend, MatchesIndependentCenteredSTFT) {
  auto samples = Signal(1281);
  OfflineStream stream(ParakeetConfig());
  stream.AcceptWaveform(16000, samples.data(), samples.size());
  auto frames = stream.GetFrames();
  ASSERT_EQ(frames.size(), 8 * 128);
  const int32_t bins[] = {0, 1, 16, 32, 64, 96, 126, 127};
  for (int32_t i = 0; i < 8; ++i) {
    for (int32_t j = 0; j < 8; ++j) {
      EXPECT_NEAR(frames[i * 128 + bins[j]], kReference[i][j], 2e-4)
          << "frame=" << i << " bin=" << bins[j];
    }
  }
}

TEST(ParakeetFrontend, ResamplesBeforeExtracting) {
  auto samples = Signal(24001);
  OfflineStream actual(ParakeetConfig());
  actual.AcceptWaveform(24000, samples.data(), samples.size());
  LinearResample resampler(24000, 16000, 0.99 * 0.5 * 16000, 6);
  std::vector<float> resampled;
  resampler.Resample(samples.data(), samples.size(), true, &resampled);
  OfflineStream expected(ParakeetConfig());
  expected.AcceptWaveform(16000, resampled.data(), resampled.size());
  EXPECT_EQ(actual.GetFrames(), expected.GetFrames());
}

TEST(ParakeetFrontend, PreservesPreemphasisAcrossInputChunks) {
  auto samples = Signal(1281);
  OfflineStream whole(ParakeetConfig());
  whole.AcceptWaveform(16000, samples.data(), samples.size());
  const auto expected = whole.GetFrames();
  for (int32_t chunk_size : {1, 159, 160, 321, 512}) {
    SCOPED_TRACE(chunk_size);
    OfflineStream chunked(ParakeetConfig());
    for (int32_t offset = 0; offset < samples.size(); offset += chunk_size) {
      const int32_t n = std::min<int32_t>(chunk_size, samples.size() - offset);
      chunked.AcceptWaveform(16000, samples.data() + offset, n);
    }
    EXPECT_EQ(chunked.GetFrames(), expected);
  }
}
}  // namespace sherpa_onnx
