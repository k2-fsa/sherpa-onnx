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

}  // namespace
}  // namespace sherpa_onnx
