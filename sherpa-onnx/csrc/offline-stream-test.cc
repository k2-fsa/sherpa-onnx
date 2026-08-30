// sherpa-onnx/csrc/offline-stream-test.cc
//
// Copyright (c)  2026  mitsu-h

#include "sherpa-onnx/csrc/offline-stream.h"

#include <string>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(OfflineRecognitionResult, SerializesNBestHypotheses) {
  OfflineRecognitionResult result;
  result.text = "top one";
  result.tokens = {"top", " one"};

  OfflineRecognitionHypothesis first;
  first.text = "top one";
  first.tokens = {"top", " one"};
  first.timestamps = {0.0f, 0.4f};
  first.ys_log_probs = {-0.1f, -0.2f};
  first.score = -0.15;

  OfflineRecognitionHypothesis second;
  second.text = "top won";
  second.tokens = {"top", " won"};
  second.timestamps = {0.0f, 0.4f};
  second.ys_log_probs = {-0.1f, -0.3f};
  second.score = -0.2;

  result.hypotheses = {first, second};

  std::string json = result.AsJsonString();
  EXPECT_NE(json.find("\"hypotheses\": [{"), std::string::npos);
  EXPECT_NE(json.find("\"text\": \"top one\""), std::string::npos);
  EXPECT_NE(json.find("\"text\": \"top won\""), std::string::npos);
  EXPECT_NE(json.find("\"score\": -0.150000"), std::string::npos);
  EXPECT_NE(json.find("\"score\": -0.200000"), std::string::npos);
}

TEST(OfflineRecognitionResult, SerializesEmptyHypothesisList) {
  OfflineRecognitionResult result;
  std::string json = result.AsJsonString();
  EXPECT_NE(json.find("\"hypotheses\": []"), std::string::npos);
}

}  // namespace sherpa_onnx
