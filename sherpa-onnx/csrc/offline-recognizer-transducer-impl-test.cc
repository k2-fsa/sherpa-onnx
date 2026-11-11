// sherpa-onnx/csrc/offline-recognizer-transducer-impl-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-recognizer-transducer-impl.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

// A line containing only an ID defines the space token with that ID.
const char *kSymbols = R"(<blk> 0
<unk> 1
你 2
好 3
4
그 5
는 6
괜 7
)";

std::string Decode(const std::vector<int64_t> &tokens, bool remove_cjk_space) {
  SymbolTable symbols(kSymbols, false);
  OfflineTransducerDecoderResult decoded;
  decoded.tokens = tokens;
  return Convert(decoded, symbols, 10, 4, remove_cjk_space).text;
}

TEST(OfflineTransducerConvert, RemoveCjkSpaceEnabled) {
  // "你 好" -> "你好"
  EXPECT_EQ(Decode({2, 4, 3}, true), "你好");
  // Korean keeps its word spaces: "그는 괜" is left untouched
  EXPECT_EQ(Decode({5, 6, 4, 7}, true), "그는 괜");
}

TEST(OfflineTransducerConvert, RemoveCjkSpaceDisabled) {
  // "你 好" keeps its space
  EXPECT_EQ(Decode({2, 4, 3}, false), "你 好");
  EXPECT_EQ(Decode({5, 6, 4, 7}, false), "그는 괜");
}

TEST(OfflineTransducerConvert, RemoveCjkSpaceDefaultsToEnabled) {
  SymbolTable symbols(kSymbols, false);
  OfflineTransducerDecoderResult decoded;
  decoded.tokens = {2, 4, 3};
  EXPECT_EQ(Convert(decoded, symbols, 10, 4).text, "你好");
}

TEST(OfflineTransducerConvert, RemoveCjkSpaceAppliesToAllHypotheses) {
  SymbolTable symbols(kSymbols, false);
  OfflineTransducerDecoderResult decoded;
  decoded.tokens = {2, 4, 3};

  OfflineTransducerDecoderHypothesis first;
  first.tokens = decoded.tokens;
  first.score = -0.1;
  OfflineTransducerDecoderHypothesis second;
  second.tokens = {3, 4, 2};
  second.score = -0.2;
  decoded.hypotheses = {first, second};

  for (int32_t remove_cjk_space : {0, 1}) {
    auto result = Convert(decoded, symbols, 10, 4, remove_cjk_space);
    ASSERT_EQ(result.hypotheses.size(), 2u);
    EXPECT_EQ(result.text, remove_cjk_space ? "你好" : "你 好");
    EXPECT_EQ(result.text, result.hypotheses[0].text);
    EXPECT_EQ(result.tokens, result.hypotheses[0].tokens);
    EXPECT_EQ(result.hypotheses[1].text,
              remove_cjk_space ? "好你" : "好 你");
    EXPECT_DOUBLE_EQ(result.hypotheses[0].score, first.score);
    EXPECT_DOUBLE_EQ(result.hypotheses[1].score, second.score);
  }
}

}  // namespace
}  // namespace sherpa_onnx
