// sherpa-onnx/csrc/online-recognizer-result-test.cc
//
// Copyright (c)  2026  Julian Pscheid

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/online-recognizer.h"

namespace sherpa_onnx {

TEST(OnlineRecognizerResult, RemoveLeadingPunctDropsPunctToken) {
  OnlineRecognizerResult r;
  r.text = ". Hello world";
  r.tokens = {".", " Hello", " world"};
  r.timestamps = {0.1, 0.5, 0.9};
  r.ys_probs = {-0.1, -0.2, -0.3};

  RemoveLeadingPunct(&r);

  EXPECT_EQ(r.text, " Hello world");
  EXPECT_EQ(r.tokens, (std::vector<std::string>{" Hello", " world"}));
  EXPECT_EQ(r.timestamps, (std::vector<float>{0.5, 0.9}));
  EXPECT_EQ(r.ys_probs, (std::vector<float>{-0.2, -0.3}));
}

TEST(OnlineRecognizerResult, RemoveLeadingPunctTrimsTokenWithText) {
  OnlineRecognizerResult r;
  r.text = ",Hello world";
  r.tokens = {",Hello", " world"};
  r.timestamps = {0.1, 0.5};

  RemoveLeadingPunct(&r);

  EXPECT_EQ(r.text, "Hello world");
  EXPECT_EQ(r.tokens, (std::vector<std::string>{"Hello", " world"}));
  EXPECT_EQ(r.timestamps, (std::vector<float>{0.1, 0.5}));
}

TEST(OnlineRecognizerResult, RemoveLeadingPunctKeepsUnrelatedTokens) {
  OnlineRecognizerResult r;
  r.text = "Hello world";
  r.tokens = {" Hello", " world"};
  r.timestamps = {0.1, 0.5};

  RemoveLeadingPunct(&r);

  EXPECT_EQ(r.text, "Hello world");
  EXPECT_EQ(r.tokens, (std::vector<std::string>{" Hello", " world"}));
  EXPECT_EQ(r.timestamps, (std::vector<float>{0.1, 0.5}));
}

TEST(OnlineRecognizerResult, RemoveLeadingPunctKeepsAllParallelFields) {
  OnlineRecognizerResult r;
  r.text = "。你好";
  r.tokens = {" ", "。", "你好"};
  r.timestamps = {0.0, 0.1, 0.5};
  r.ys_probs = {-0.1, -0.2, -0.3};
  r.lm_probs = {-1.1, -1.2, -1.3};
  r.context_scores = {0.1, 0.2, 0.3};
  r.words = {42, 901};

  RemoveLeadingPunct(&r);

  EXPECT_EQ(r.text, "你好");
  EXPECT_EQ(r.tokens, (std::vector<std::string>{" ", "你好"}));
  EXPECT_EQ(r.timestamps, (std::vector<float>{0.0, 0.5}));
  EXPECT_EQ(r.ys_probs, (std::vector<float>{-0.1, -0.3}));
  EXPECT_EQ(r.lm_probs, (std::vector<float>{-1.1, -1.3}));
  EXPECT_EQ(r.context_scores, (std::vector<float>{0.1, 0.3}));
  // words are word IDs, not token positions.
  EXPECT_EQ(r.words, (std::vector<int32_t>{42, 901}));
}

TEST(OnlineRecognizerResult, RemoveLeadingPunctSkipsNonParallelFields) {
  OnlineRecognizerResult r;
  r.text = ".";
  r.tokens = {"."};
  r.timestamps = {0.1};
  r.ys_probs = {-0.1, -0.2};

  RemoveLeadingPunct(&r);

  EXPECT_EQ(r.text, "");
  EXPECT_TRUE(r.tokens.empty());
  EXPECT_TRUE(r.timestamps.empty());
  EXPECT_EQ(r.ys_probs, (std::vector<float>{-0.1, -0.2}));
}

TEST(OnlineRecognizerResult, RemoveLeadingPunctHandlesEmptyResult) {
  OnlineRecognizerResult r;

  RemoveLeadingPunct(&r);

  EXPECT_EQ(r.text, "");
  EXPECT_TRUE(r.tokens.empty());
}

TEST(OnlineRecognizerResult, RemoveLeadingPunctLeavesByteTokensAlone) {
  // Byte fallback: the punctuation is spread over byte tokens, so only
  // the text changes.
  OnlineRecognizerResult r;
  r.text = "。你好";
  r.tokens = {"<0xE3>", "<0x80>", "<0x82>", "你好"};
  r.timestamps = {0.1, 0.1, 0.1, 0.5};

  RemoveLeadingPunct(&r);

  EXPECT_EQ(r.text, "你好");
  EXPECT_EQ(r.tokens, (std::vector<std::string>{"<0xE3>", "<0x80>", "<0x82>",
                                                 "你好"}));
  EXPECT_EQ(r.timestamps, (std::vector<float>{0.1, 0.1, 0.1, 0.5}));
}

}  // namespace sherpa_onnx
