// sherpa-onnx/csrc/offline-stream-test.cc
//
// Copyright (c)  2026  Javier Martinez

#include "sherpa-onnx/csrc/offline-stream.h"

#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(OfflineStream, SpeakerDiarizationKeepsSamples) {
  std::vector<float> samples = {0.1, -0.2, 0.3, -0.4, 0.5};

  OfflineStream stream(SpeakerDiarizationTag{});
  stream.AcceptWaveform(16000, samples.data(), samples.size());

  EXPECT_EQ(stream.GetFrames(), samples);
}

TEST(OfflineStream, SpeakerDiarizationResamples) {
  std::vector<float> samples(8000, 0.1);

  OfflineStream stream(SpeakerDiarizationTag{});
  stream.AcceptWaveform(8000, samples.data(), samples.size());

  EXPECT_NEAR(stream.GetFrames().size(), 16000, 10);
}

}  // namespace sherpa_onnx
