// sherpa-onnx/csrc/text2token-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <fstream>
#include <sstream>
#include <string>

#include "cppinyin/csrc/cppinyin.h"
#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/utils.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_onnx {

// Please refer to
// https://github.com/pkufool/sherpa-test-data
// to download test data for testing
static const char dir[] = "/tmp/sherpa-test-data";

TEST(TEXT2TOKEN, TEST_cjkchar) {
  std::ostringstream oss;
  oss << dir << "/text2token/tokens_cn.txt";

  std::string tokens = oss.str();

  if (!std::ifstream(tokens).good()) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TEST_cjkchar()."
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git "
        "/tmp/sherpa-test-data");
    return;
  }

  auto sym_table = SymbolTable(tokens);

  std::string text =
      "世界人民大团结\n中国 V S 美国\n\n";  // Test blank lines also

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;

  auto r = EncodeHotwords(iss, "cjkchar", sym_table, nullptr, &ids, &scores);

  std::vector<std::vector<int32_t>> expected_ids(
      {{379, 380, 72, 874, 93, 1251, 489}, {262, 147, 3423, 2476, 21, 147}});
  EXPECT_EQ(ids, expected_ids);

  EXPECT_EQ(scores.size(), 0);
}

TEST(TEXT2TOKEN, TEST_bpe) {
  std::ostringstream oss;
  oss << dir << "/text2token/tokens_en.txt";
  std::string tokens = oss.str();
  oss.clear();
  oss.str("");
  oss << dir << "/text2token/bpe_en.vocab";
  std::string bpe = oss.str();
  if (!std::ifstream(tokens).good() || !std::ifstream(bpe).good()) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TEST_bpe()."
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git "
        "/tmp/sherpa-test-data");
    return;
  }

  auto sym_table = SymbolTable(tokens);
  auto bpe_processor = std::make_unique<ssentencepiece::Ssentencepiece>(bpe);

  std::string text = "HELLO WORLD\nI LOVE YOU :2.0";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;

  auto r =
      EncodeHotwords(iss, "bpe", sym_table, bpe_processor.get(), &ids, &scores);

  std::vector<std::vector<int32_t>> expected_ids(
      {{22, 58, 24, 425}, {19, 370, 47}});
  EXPECT_EQ(ids, expected_ids);

  std::vector<float> expected_scores({0, 2.0});
  EXPECT_EQ(scores, expected_scores);
}

TEST(TEXT2TOKEN, TEST_cjkchar_bpe) {
  std::ostringstream oss;
  oss << dir << "/text2token/tokens_mix.txt";
  std::string tokens = oss.str();
  oss.clear();
  oss.str("");
  oss << dir << "/text2token/bpe_mix.vocab";
  std::string bpe = oss.str();
  if (!std::ifstream(tokens).good() || !std::ifstream(bpe).good()) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TEST_cjkchar_bpe()."
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git "
        "/tmp/sherpa-test-data");
    return;
  }

  auto sym_table = SymbolTable(tokens);
  auto bpe_processor = std::make_unique<ssentencepiece::Ssentencepiece>(bpe);

  std::string text = "世界人民 GOES TOGETHER :1.5\n中国 GOES WITH 美国 :0.5";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;

  auto r = EncodeHotwords(iss, "cjkchar+bpe", sym_table, bpe_processor.get(),
                          &ids, &scores);

  std::vector<std::vector<int32_t>> expected_ids(
      {{1368, 1392, 557, 680, 275, 178, 475},
       {685, 736, 275, 178, 179, 921, 736}});
  EXPECT_EQ(ids, expected_ids);

  std::vector<float> expected_scores({1.5, 0.5});
  EXPECT_EQ(scores, expected_scores);
}

TEST(TEXT2TOKEN, TEST_bbpe) {
  std::ostringstream oss;
  oss << dir << "/text2token/tokens_bbpe.txt";
  std::string tokens = oss.str();
  oss.clear();
  oss.str("");
  oss << dir << "/text2token/bbpe.vocab";
  std::string bpe = oss.str();
  if (!std::ifstream(tokens).good() || !std::ifstream(bpe).good()) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TEST_bbpe()."
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git "
        "/tmp/sherpa-test-data");
    return;
  }

  auto sym_table = SymbolTable(tokens);
  auto bpe_processor = std::make_unique<ssentencepiece::Ssentencepiece>(bpe);

  std::string text = "频繁 :1.0\n李鞑靼";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;

  auto r =
      EncodeHotwords(iss, "bpe", sym_table, bpe_processor.get(), &ids, &scores);

  std::vector<std::vector<int32_t>> expected_ids(
      {{259, 1118, 234, 188, 132}, {259, 1585, 236, 161, 148, 236, 160, 191}});
  EXPECT_EQ(ids, expected_ids);

  std::vector<float> expected_scores({1.0, 0});
  EXPECT_EQ(scores, expected_scores);
}

TEST(TEXT2TOKEN, TEST_keyword_bpe) {
  std::ostringstream oss;
  oss << dir << "/text2token/tokens_en.txt";
  std::string tokens = oss.str();
  oss.clear();
  oss.str("");
  oss << dir << "/text2token/bpe_en.vocab";
  std::string bpe = oss.str();
  if (!std::ifstream(tokens).good() || !std::ifstream(bpe).good()) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TEST_bpe()."
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git "
        "/tmp/sherpa-test-data");
    return;
  }

  auto sym_table = SymbolTable(tokens);
  auto bpe_processor = std::make_unique<ssentencepiece::Ssentencepiece>(bpe);

  std::string text = "HELLO WORLD :2.0 @FUCK_WORLD\nI LOVE YOU #0.25";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<std::string> keywords;
  std::vector<float> scores;
  std::vector<float> thresholds;

  auto r = EncodeKeywords(iss, "bpe", sym_table, bpe_processor.get(), nullptr,
                          &ids, &keywords, &scores, &thresholds);

  std::vector<std::vector<int32_t>> expected_ids(
      {{22, 58, 24, 425}, {19, 370, 47}});
  EXPECT_EQ(ids, expected_ids);

  std::vector<std::string> expected_keywords({"FUCK WORLD", "I LOVE YOU"});
  EXPECT_EQ(keywords, expected_keywords);

  std::vector<float> expected_scores({2.0, 0});
  EXPECT_EQ(scores, expected_scores);

  std::vector<float> expected_thresholds({0, 0.25});
  EXPECT_EQ(thresholds, expected_thresholds);
}

TEST(TEXT2TOKEN, TEST_keyword_ppinyin) {
  std::ostringstream oss;
  oss << dir << "/text2token/tokens_pinyin.txt";
  std::string tokens = oss.str();
  oss.clear();
  oss.str("");
  oss << dir << "/text2token/pinyin.raw";
  std::string lexicon = oss.str();
  if (!std::ifstream(tokens).good() || !std::ifstream(lexicon).good()) {
    SHERPA_ONNX_LOGE(
        "No test data found, skipping TEST_bpe()."
        "You can download the test data by: "
        "git clone https://github.com/pkufool/sherpa-test-data.git "
        "/tmp/sherpa-test-data");
    return;
  }

  auto sym_table = SymbolTable(tokens);
  auto py_encoder = std::make_unique<cppinyin::PinyinEncoder>(lexicon);

  std::string text = "世界人民大团结 :2.0\n美国大选 #0.4\n中国很美 @中国很漂亮";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<std::string> keywords;
  std::vector<float> scores;
  std::vector<float> thresholds;

  auto r = EncodeKeywords(iss, "ppinyin", sym_table, nullptr, py_encoder.get(),
                          &ids, &keywords, &scores, &thresholds);

  std::vector<std::string> expected_keywords(
      {"世界人民大团结", "美国大选", "中国很漂亮"});
  EXPECT_EQ(keywords, expected_keywords);

  std::vector<float> expected_scores({2.0, 0, 0});
  EXPECT_EQ(scores, expected_scores);

  std::vector<float> expected_thresholds({0, 0.4, 0});
  EXPECT_EQ(thresholds, expected_thresholds);

  std::vector<std::vector<int32_t>> expected_ids(
      {{13, 36, 24, 155, 39, 41, 58, 137, 53, 71, 77, 114, 24, 138},
       {58, 125, 43, 66, 53, 71, 48, 44},
       {10, 50, 43, 66, 75, 148, 58, 125}});
  EXPECT_EQ(ids, expected_ids);
}

}  // namespace sherpa_onnx
