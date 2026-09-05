// sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl-test.cc
//
// Copyright (c)  2026  kyo-zzz

#include "sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.h"

#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

// Digital silence produces constant fbank frames (dither is disabled for this
// model family), so FunASRNanoAudioIsSilent() must report silence for them
// and signal for any varying content. DecodeStreams() uses this to return an
// empty transcript before hotwords/language prompt tokens can bias the LLM
// decoder into hallucinating text for silent audio, mirroring the Qwen3-ASR
// recognizer.
TEST(FunASRNanoAudioIsSilent, ConstantFramesAreSilence) {
  // LFR stacks feature_dim*window floats per output frame; the values are
  // identical for a constant input.
  std::vector<float> features = {2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f};
  EXPECT_TRUE(FunASRNanoAudioIsSilent(features.data(), features.size()));
}

TEST(FunASRNanoAudioIsSilent, RepeatedNonUniformFramesAreSignal) {
  // alternating values still vary frame to frame, so this is signal
  std::vector<float> features = {1.0f, 2.0f, 1.0f, 2.0f};
  EXPECT_FALSE(FunASRNanoAudioIsSilent(features.data(), features.size()));
}

TEST(FunASRNanoAudioIsSilent, VaryingFramesAreSignal) {
  std::vector<float> features = {2.5f, 2.5f, -1.25f, 2.5f, 2.5f, 2.5f};
  EXPECT_FALSE(FunASRNanoAudioIsSilent(features.data(), features.size()));
}

TEST(FunASRNanoAudioIsSilent, ZerosAreSilence) {
  std::vector<float> features(64, 0.0f);
  EXPECT_TRUE(FunASRNanoAudioIsSilent(features.data(), features.size()));
}

TEST(FunASRNanoAudioIsSilent, SingleValueIsSilence) {
  // A single frame carries no variation to inspect.
  std::vector<float> features = {2.5f};
  EXPECT_TRUE(FunASRNanoAudioIsSilent(features.data(), features.size()));
}

TEST(FunASRNanoAudioIsSilent, EmptyInputIsNotSilence) {
  // An empty input is handled by the caller (num_frames <= 0), not here.
  float unused = 0;
  EXPECT_FALSE(FunASRNanoAudioIsSilent(&unused, 0));
}

}  // namespace sherpa_onnx
