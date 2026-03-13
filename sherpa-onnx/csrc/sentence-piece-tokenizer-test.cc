// sherpa-onnx/csrc/sentence-piece-tokenizer-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include "sherpa-onnx/csrc/sentence-piece-tokenizer.h"

#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

static const char dir[] = "/tmp/sherpa-onnx-test-data";

TEST(SpTokenizer, TestEncode) {
  auto vocab_json = std::string(dir) + "/vocab.json";
  auto token_scores_json = std::string(dir) + "/token_scores.json";

  if (!std::ifstream(vocab_json).good() ||
      !std::ifstream(token_scores_json).good()) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TestEncode()."
        "You can download the test data from: "
        "https://huggingface.co/csukuangfj/sherpa-onnx-test-data/tree/main"
        "and put it inside "
        "/tmp/sherpa-onnx-test-data");
    return;
  }

  auto sp = SentencePieceTokenizer(vocab_json, token_scores_json);
  std::string text =
      "How are you doing today? Fantastic! How about you? I am OK.";
  std::vector<std::string> expected_tokens = {
      "▁How", "▁are", "▁you",   "▁doing", "▁today", "?",
      "▁F",   "an",   "tastic", "!",      "▁How",   "▁about",
      "▁you", "?",    "▁I",     "▁am",    "▁OK",    "."};

  std::vector<std::string> tokens = sp.EncodeTokens(text);
  EXPECT_EQ(tokens, expected_tokens);

  std::vector<int32_t> expected_ids = {668, 304, 270,  473, 630,  292,
                                       496, 456, 2264, 682, 668,  315,
                                       270, 292, 268,  686, 1183, 263};

  std::vector<int32_t> token_ids = sp.EncodeIds(text);
  EXPECT_EQ(token_ids, expected_ids);
}

}  // namespace sherpa_onnx
