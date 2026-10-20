// sherpa-onnx/csrc/recognizer-result-json-test.cc
//
// Copyright (c)  2026  hulkbig

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/online-recognizer.h"

namespace sherpa_onnx {
namespace {

std::vector<std::string> GetJsonStrings() {
  std::vector<std::string> strings = {
      "", "plain text", "\"quoted\" \\ path /", "你好世界 🌍"};
  std::string all_controls;
  for (int32_t c = 0; c < 0x20; ++c) {
    strings.emplace_back(1, static_cast<char>(c));
    all_controls.push_back(static_cast<char>(c));
  }
  strings.push_back("before" + all_controls + "after");
  return strings;
}

class RecognizerResultJsonTest : public ::testing::TestWithParam<std::string> {
};

TEST_P(RecognizerResultJsonTest, OnlineText) {
  OnlineRecognizerResult result;
  result.text = GetParam();
  auto json = nlohmann::json::parse(result.AsJsonString(), nullptr, false);
  ASSERT_FALSE(json.is_discarded());
  EXPECT_EQ(json["text"].get<std::string>(), GetParam());
}

TEST_P(RecognizerResultJsonTest, OnlineTokens) {
  OnlineRecognizerResult result;
  result.tokens = {"first", GetParam(), "last"};
  auto json = nlohmann::json::parse(result.AsJsonString(), nullptr, false);
  ASSERT_FALSE(json.is_discarded());
  EXPECT_EQ(json["tokens"].get<std::vector<std::string>>(), result.tokens);
}

TEST_P(RecognizerResultJsonTest, OfflineTextAndMetadata) {
  OfflineRecognitionResult result;
  result.text = GetParam();
  result.lang = GetParam();
  result.emotion = GetParam();
  result.event = GetParam();
  auto json = nlohmann::json::parse(result.AsJsonString(), nullptr, false);
  ASSERT_FALSE(json.is_discarded());
  for (const auto *field : {"text", "lang", "emotion", "event"}) {
    EXPECT_EQ(json[field].get<std::string>(), GetParam()) << field;
  }
}

TEST_P(RecognizerResultJsonTest, OfflineTokens) {
  OfflineRecognitionResult result;
  result.tokens = {"first", GetParam(), "last"};
  auto json = nlohmann::json::parse(result.AsJsonString(), nullptr, false);
  ASSERT_FALSE(json.is_discarded());
  EXPECT_EQ(json["tokens"].get<std::vector<std::string>>(), result.tokens);
}

TEST_P(RecognizerResultJsonTest, OfflineSegmentTexts) {
  OfflineRecognitionResult result;
  result.segment_timestamps = {0.0f, 1.0f, 2.0f};
  result.segment_durations = {1.0f, 1.0f, 1.0f};
  result.segment_texts = {"first", GetParam(), "last"};
  auto json = nlohmann::json::parse(result.AsJsonString(), nullptr, false);
  ASSERT_FALSE(json.is_discarded());
  EXPECT_EQ(json["segment_texts"].get<std::vector<std::string>>(),
            result.segment_texts);
}

INSTANTIATE_TEST_SUITE_P(Strings, RecognizerResultJsonTest,
                         ::testing::ValuesIn(GetJsonStrings()));

TEST(RecognizerResultJson, OfflineByteTokens) {
  OfflineRecognitionResult result;
  std::vector<std::string> expected;
  constexpr char hex[] = "0123456789ABCDEF";
  for (int32_t c = 0x80; c <= 0xff; ++c) {
    result.tokens.emplace_back(1, static_cast<char>(c));
    std::string token = "<0x";
    token += hex[c >> 4];
    token += hex[c & 0xf];
    expected.push_back(token + ">");
  }
  result.tokens.push_back("你好 🌍");
  expected.push_back(result.tokens.back());
  auto json = nlohmann::json::parse(result.AsJsonString(), nullptr, false);
  ASSERT_FALSE(json.is_discarded());
  EXPECT_EQ(json["tokens"].get<std::vector<std::string>>(), expected);
}

TEST(RecognizerResultJson, EmptyResults) {
  auto online = nlohmann::json::parse(OnlineRecognizerResult{}.AsJsonString());
  EXPECT_EQ(online["text"], "");
  EXPECT_EQ(online["tokens"], nlohmann::json::array());
  EXPECT_EQ(online["timestamps"], nlohmann::json::array());
  EXPECT_EQ(online["segment"], 0);
  EXPECT_EQ(online["start_time"], 0);
  EXPECT_EQ(online["is_final"], false);
  EXPECT_EQ(online["is_eof"], false);

  auto offline =
      nlohmann::json::parse(OfflineRecognitionResult{}.AsJsonString());
  for (const auto *field : {"text", "lang", "emotion", "event"}) {
    EXPECT_EQ(offline[field], "") << field;
  }
  EXPECT_EQ(offline["tokens"], nlohmann::json::array());
  EXPECT_EQ(offline["timestamps"], nlohmann::json::array());
  EXPECT_FALSE(offline.contains("segment_texts"));
}

}  // namespace
}  // namespace sherpa_onnx
