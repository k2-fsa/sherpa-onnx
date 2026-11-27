// sherpa-onnx/csrc/online-stream-test.cc
//
// Copyright (c)  2026  Julian Pscheid

#include "sherpa-onnx/csrc/online-stream.h"

#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

TEST(OnlineStream, GigaAMV4LogMel) {
  FeatureExtractorConfig config;
  config.feature_dim = 64;
  config.low_freq = 0;
  config.high_freq = 8000;
  config.snip_edges = true;
  config.frame_length_ms = 20;
  config.remove_dc_offset = false;
  config.preemph_coeff = 0;
  config.window_type = "hann";
  config.round_to_power_of_two = false;
  config.is_gigaam = true;

  OnlineStream stream(config);
  std::vector<float> samples(480, 0.0f);
  samples[80] = 1.0f;
  stream.AcceptWaveform(16000, samples.data(), samples.size());
  ASSERT_EQ(stream.NumFramesReady(), 2);
  const auto frames = stream.GetFrames(0, 2);
  ASSERT_EQ(frames.size(), 128);

  // The windowed impulse has power 1/4 at every frequency. With the
  // reference GigaAM filterbank, each log-mel band is log(sum(weights)/4).
  EXPECT_NEAR(frames[0], -2.883459, 1e-4);
  EXPECT_NEAR(frames[31], -0.759192, 1e-4);
  EXPECT_NEAR(frames[63], 0.478701, 1e-4);
  for (int32_t i = 64; i < 128; ++i) {
    EXPECT_NEAR(frames[i], std::log(1e-9f), 1e-5);
  }
}

TEST(OnlineStream, NemoAdditiveLogGuard) {
  std::vector<float> audio(16000, 0.0f);
  for (bool nemo_buffered : {false, true}) {
    FeatureExtractorConfig config;
    config.nemo_buffered_capacity = nemo_buffered ? 672 : 0;
    OnlineStream stream(config);
    stream.AcceptWaveform(16000, audio.data(), audio.size());
    EXPECT_TRUE(stream.GetFrames(0, 0).empty());
    auto frames = stream.GetFrames(0, 2);
    float epsilon = std::numeric_limits<float>::epsilon();
    float expected = std::log(nemo_buffered ? epsilon / 2 : epsilon);
    for (float value : frames) {
      EXPECT_NEAR(value, expected, 1e-5);
    }
  }
}

TEST(OnlineStream, StreamingNemoAdditiveLogGuard) {
  // Set by the streaming NeMo CTC and transducer recognizers
  FeatureExtractorConfig config;
  config.use_nemo_log_guard = true;
  OnlineStream stream(config);
  std::vector<float> audio(16000, 0.0f);
  stream.AcceptWaveform(16000, audio.data(), audio.size());
  float expected = std::log(std::numeric_limits<float>::epsilon() / 2);
  for (float value : stream.GetFrames(0, 2)) {
    EXPECT_NEAR(value, expected, 1e-5);
  }
}

TEST(OnlineStream, NemoFlushesResamplerTail) {
  FeatureExtractorConfig config;
  config.nemo_buffered_capacity = 16;
  OnlineStream stream(config);
  std::vector<float> audio(4803);
  for (int32_t i = 0; i < audio.size(); ++i) {
    audio[i] = 0.1 * std::sin(0.037 * i);
  }
  stream.AcceptWaveform(48000, audio.data(), 2401);
  stream.AcceptWaveform(48000, audio.data() + 2401, audio.size() - 2401);
  EXPECT_EQ(stream.NumFramesReady(), 9);
  stream.InputFinished();
  EXPECT_EQ(stream.NumFramesReady(), 10);
  EXPECT_EQ(stream.NumTrailingSamples(), 1);

  LinearResample resampler(48000, 16000, 0.99 * 0.5 * 16000, 6);
  std::vector<float> resampled;
  resampler.Resample(audio.data(), 2401, false, &resampled);
  std::vector<float> last;
  resampler.Resample(audio.data() + 2401, audio.size() - 2401, true, &last);
  resampled.insert(resampled.end(), last.begin(), last.end());
  ASSERT_EQ(resampled.size(), 1601);
  OnlineStream reference(config);
  reference.AcceptWaveform(16000, resampled.data(), resampled.size());
  reference.InputFinished();
  EXPECT_EQ(stream.GetBufferedFeatures(0, 10, true, nullptr),
            reference.GetBufferedFeatures(0, 10, true, nullptr));
}

TEST(OnlineStream, LastFrameAfterReset) {
  OnlineStream stream(FeatureExtractorConfig{});
  std::vector<float> audio(16000, 0.1f);
  stream.AcceptWaveform(16000, audio.data(), audio.size());
  stream.GetNumProcessedFrames() = 40;
  stream.Reset();
  stream.InputFinished();
  int32_t ready = stream.NumFramesReady();
  ASSERT_GT(ready, 0);
  EXPECT_TRUE(stream.IsLastFrame(ready - 1));
  EXPECT_FALSE(stream.IsLastFrame(ready - 2));
}

TEST(OnlineStream, RetainedLeftContextAfterReset) {
  OnlineStream stream(FeatureExtractorConfig{});
  std::vector<float> audio(16000, 0.1f);
  stream.AcceptWaveform(16000, audio.data(), audio.size());
  auto before = stream.GetFrames(10, 30);
  stream.GetNumProcessedFrames() = 40;
  stream.Reset();
  auto after = stream.GetFrames(-30, 30);
  EXPECT_EQ(before, after);
  EXPECT_EQ(stream.GetNumFramesSinceStart(), 40);
  EXPECT_EQ(stream.GetNumProcessedFrames(), 0);
}

}  // namespace sherpa_onnx
