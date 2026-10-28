// sherpa-onnx/csrc/offline-recognizer-ctc-impl-test.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/offline-recognizer-ctc-impl.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

// Exercise the production greedy decoder, including blank removal and repeated
// token collapse, without loading an acoustic model.
OfflineCtcDecoderResult Decode(const std::vector<int32_t> &frame_tokens,
                               int32_t vocab_size) {
  std::vector<float> log_probs(frame_tokens.size() * vocab_size,
                               std::log(0.2f / (vocab_size - 1)));
  for (size_t t = 0; t != frame_tokens.size(); ++t) {
    log_probs[t * vocab_size + frame_tokens[t]] = std::log(0.8f);
  }

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<int64_t, 3> shape = {1, static_cast<int64_t>(frame_tokens.size()),
                                  vocab_size};
  std::array<int64_t, 1> length_shape = {1};
  int64_t length = frame_tokens.size();
  auto probs = Ort::Value::CreateTensor<float>(memory_info, log_probs.data(),
                                               log_probs.size(), shape.data(),
                                               shape.size());
  auto lengths = Ort::Value::CreateTensor<int64_t>(
      memory_info, &length, 1, length_shape.data(), length_shape.size());
  OfflineCtcGreedySearchDecoder decoder(0);
  auto results = decoder.Decode(std::move(probs), std::move(lengths));
  EXPECT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].tokens.size(), results[0].timestamps.size());
  return std::move(results[0]);
}

void ExpectResult(const OfflineRecognitionResult &result,
                  const std::vector<std::string> &tokens,
                  const std::vector<float> &timestamps,
                  const std::string &text) {
  EXPECT_EQ(result.tokens, tokens);
  EXPECT_EQ(result.text, text);
  EXPECT_EQ(result.timestamps.size(), result.tokens.size());
  ASSERT_EQ(result.timestamps.size(), timestamps.size())
      << "Actual timestamps: " << ::testing::PrintToString(result.timestamps);
  for (size_t i = 0; i != timestamps.size(); ++i) {
    EXPECT_FLOAT_EQ(result.timestamps[i], timestamps[i]);
  }
}

TEST(OfflineCtcConvert, FiltersSilenceTimestamps) {
  SymbolTable symbols("<eps> 0\nSIL 1\nA 2\nB 3\n", false);
  auto decoded = Decode({1, 1, 2, 0, 1, 0, 3, 3}, 4);
  EXPECT_EQ(decoded.tokens, (std::vector<int64_t>{1, 2, 1, 3}));
  EXPECT_EQ(decoded.timestamps, (std::vector<int32_t>{0, 2, 4, 6}));

  ExpectResult(Convert(decoded, symbols, 10, 1), {"A", "B"}, {0.02f, 0.06f},
               "AB");
}

TEST(OfflineCtcConvert, FiltersEndOfSentenceTimestamp) {
  SymbolTable symbols("<blk> 0\nA 1\nB 2\n</s> 3\n", false);
  auto decoded = Decode({1, 1, 0, 2, 0, 3, 3}, 4);
  EXPECT_EQ(decoded.tokens, (std::vector<int64_t>{1, 2, 3}));
  EXPECT_EQ(decoded.timestamps, (std::vector<int32_t>{0, 3, 5}));

  ExpectResult(Convert(decoded, symbols, 10, 4), {"A", "B"}, {0.0f, 0.12f},
               "AB");
}

TEST(OfflineCtcConvert, AllFilteredTokensHaveNoTimestamps) {
  SymbolTable symbols("<eps> 0\nSIL 1\n</s> 2\n", false);
  auto decoded = Decode({1, 1, 0, 2, 2}, 3);
  EXPECT_EQ(decoded.tokens, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(decoded.timestamps, (std::vector<int32_t>{0, 3}));

  ExpectResult(Convert(decoded, symbols, 10, 4), {}, {}, "");
}

TEST(OfflineCtcConvert, KeepsOrdinaryTokensAndTimestamps) {
  SymbolTable symbols("<blk> 0\nA 1\nB 2\n", false);
  auto decoded = Decode({0, 1, 1, 0, 2, 2, 0, 1}, 3);
  EXPECT_EQ(decoded.tokens, (std::vector<int64_t>{1, 2, 1}));
  EXPECT_EQ(decoded.timestamps, (std::vector<int32_t>{1, 4, 7}));

  ExpectResult(Convert(decoded, symbols, 10, 1), {"A", "B", "A"},
               {0.01f, 0.04f, 0.07f}, "ABA");
}

TEST(OfflineCtcConvert, PreservesFrameShiftAndSubsampling) {
  SymbolTable symbols("<eps> 0\nSIL 1\nA 2\nB 3\n", false);
  auto decoded = Decode({1, 1, 2, 0, 1, 0, 3, 3}, 4);

  ExpectResult(Convert(decoded, symbols, 10, 4), {"A", "B"}, {0.08f, 0.24f},
               "AB");
  ExpectResult(Convert(decoded, symbols, 20, 2), {"A", "B"}, {0.08f, 0.24f},
               "AB");
}

TEST(OfflineCtcConvert, BlankOnlyOutputStaysEmpty) {
  SymbolTable symbols("<eps> 0\nSIL 1\nA 2\n", false);
  auto decoded = Decode({0, 0, 0}, 3);
  EXPECT_TRUE(decoded.tokens.empty());
  EXPECT_TRUE(decoded.timestamps.empty());

  ExpectResult(Convert(decoded, symbols, 10, 4), {}, {}, "");
}

}  // namespace
}  // namespace sherpa_onnx
