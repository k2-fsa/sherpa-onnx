// sherpa-onnx/csrc/offline-recognizer-canary-impl-test.cc
//
// Copyright (c)  2026  kyo-zzz

#include "sherpa-onnx/csrc/offline-recognizer-canary-impl.h"

#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

// Regression test for https://github.com/k2-fsa/sherpa-onnx/issues/3919
//
// When the argmax of the first decoder logits is eos, the greedy decoder used
// to return an empty transcript: the loop broke immediately and the trailing
// pop_back() removed the only generated token. For audio that contains
// signal, that eos is a numerical near-tie and the transcript is lost
// silently. SelectCanaryFirstToken() must fall back to the best non-eos
// token in that case, and must keep eos for silent audio, where an empty
// transcript is the model's correct answer.
TEST(SelectCanaryFirstToken, SuppressesEosOnNonSilentAudio) {
  // eos_id = 2 is the argmax; token 1 is the best non-eos token
  std::vector<float> logits = {1.0f, 5.0f, 9.0f, 3.0f};
  EXPECT_EQ(SelectCanaryFirstToken(logits.data(), logits.size(), 2, true), 1);
}

TEST(SelectCanaryFirstToken, KeepsEosOnSilentAudio) {
  std::vector<float> logits = {1.0f, 5.0f, 9.0f, 3.0f};
  EXPECT_EQ(SelectCanaryFirstToken(logits.data(), logits.size(), 2, false), 2);
}

TEST(SelectCanaryFirstToken, KeepsNonEosArgmax) {
  std::vector<float> logits = {1.0f, 9.0f, 5.0f, 3.0f};
  EXPECT_EQ(SelectCanaryFirstToken(logits.data(), logits.size(), 2, true), 1);
  EXPECT_EQ(SelectCanaryFirstToken(logits.data(), logits.size(), 2, false), 1);
}

TEST(SelectCanaryFirstToken, EosAtIndexOfZero) {
  std::vector<float> logits = {9.0f, 1.0f, 5.0f};
  EXPECT_EQ(SelectCanaryFirstToken(logits.data(), logits.size(), 0, true), 2);
}

TEST(SelectCanaryFirstToken, PicksFirstTokenOnNonEosTie) {
  // tokens 1 and 2 tie; max_element returns the lower index
  std::vector<float> logits = {5.0f, 3.0f, 3.0f, 1.0f};
  EXPECT_EQ(SelectCanaryFirstToken(logits.data(), logits.size(), 0, true), 1);
}

TEST(SelectCanaryFirstToken, EosOnlyVocab) {
  // there is no non-eos token to fall back to
  std::vector<float> logits = {9.0f};
  EXPECT_EQ(SelectCanaryFirstToken(logits.data(), logits.size(), 0, true), 0);
}

// Digital silence normalized features stay below kCanarySilenceFeatureAbsMax
// while any audio with content is well above it.
TEST(CanaryHasSignal, SilenceVsSignal) {
  float unused = 0;
  EXPECT_FALSE(CanaryHasSignal(&unused, 0));

  std::vector<float> silence = {0.0f, 0.05f, -0.087f, 0.02f};
  EXPECT_FALSE(CanaryHasSignal(silence.data(), silence.size()));

  std::vector<float> signal = {0.0f, 0.05f, -1.5f, 0.02f};
  EXPECT_TRUE(CanaryHasSignal(signal.data(), signal.size()));
}

}  // namespace sherpa_onnx
