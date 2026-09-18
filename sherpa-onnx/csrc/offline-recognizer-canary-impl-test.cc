// sherpa-onnx/csrc/offline-recognizer-canary-impl-test.cc
//
// Copyright (c)  2026  kyo-zzz

#include <unordered_map>

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

TEST(DeriveCanaryLang2Id, TwoLetterCodesOnly) {
  std::unordered_map<std::string, int32_t> sym2id = {
      {"<|en|>", 90},  {"<|es|>", 84},  {"<|de|>", 78},  {"<|fr|>", 71},
      {"<|it|>", 99},  {"<|pt|>", 95},
      // longer bracketed specials are not languages
      {"<|pnc|>", 50},  {"<|noitn|>", 51}, {"<|0.00|>", 60},
      {"<|endoftext|>", 61}, {"<|startoftranscript|>", 62},
      {"hello", 1},
  };
  auto lang2id = DeriveCanaryLang2Id(sym2id);
  EXPECT_EQ(lang2id.size(), 6u);
  EXPECT_EQ(lang2id.at("en"), 90);
  EXPECT_EQ(lang2id.at("it"), 99);
  EXPECT_EQ(lang2id.at("fr"), 71);
  EXPECT_EQ(lang2id.count("pc"), 0u);
}

TEST(DeriveCanaryLang2Id, EmptyWhenNoLanguageTokens) {
  std::unordered_map<std::string, int32_t> sym2id = {{"a", 0}, {"<|pnc|>", 3}};
  EXPECT_TRUE(DeriveCanaryLang2Id(sym2id).empty());
}

TEST(ResolveCanaryLang, PassthroughFallbackAndFirstLanguage) {
  std::unordered_map<std::string, int32_t> lang2id = {
      {"en", 10}, {"it", 99}};
  EXPECT_EQ(ResolveCanaryLang(lang2id, "it", "src"), 99);
  EXPECT_EQ(ResolveCanaryLang(lang2id, "", "src"), 10);      // silent en
  EXPECT_EQ(ResolveCanaryLang(lang2id, "xx", "src"), 10);     // warn, en

  // With no en in the vocab the fallback is deterministic: the language
  // with the lowest token id.
  std::unordered_map<std::string, int32_t> no_en = {{"it", 99}, {"de", 78}};
  EXPECT_EQ(ResolveCanaryLang(no_en, "xx", "tgt"), 78);
}

}  // namespace sherpa_onnx
