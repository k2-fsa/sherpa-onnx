// sherpa-onnx/csrc/text2token-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <fstream>
#include <sstream>
#include <string>

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

  std::string text = "世界人民大团结\n中国 V S 美国";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;

  auto r = EncodeHotwords(iss, "cjkchar", sym_table, nullptr, &ids);

  std::vector<std::vector<int32_t>> expected_ids(
      {{379, 380, 72, 874, 93, 1251, 489}, {262, 147, 3423, 2476, 21, 147}});
  EXPECT_EQ(ids, expected_ids);
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

  std::string text = "HELLO WORLD\nI LOVE YOU";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;

  auto r = EncodeHotwords(iss, "bpe", sym_table, bpe_processor.get(), &ids);

  std::vector<std::vector<int32_t>> expected_ids(
      {{22, 58, 24, 425}, {19, 370, 47}});
  EXPECT_EQ(ids, expected_ids);
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

  std::string text = "世界人民 GOES TOGETHER\n中国 GOES WITH 美国";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;

  auto r =
      EncodeHotwords(iss, "cjkchar+bpe", sym_table, bpe_processor.get(), &ids);

  std::vector<std::vector<int32_t>> expected_ids(
      {{1368, 1392, 557, 680, 275, 178, 475},
       {685, 736, 275, 178, 179, 921, 736}});
  EXPECT_EQ(ids, expected_ids);
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

  std::string text = "频繁\n李鞑靼";

  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;

  auto r = EncodeHotwords(iss, "bpe", sym_table, bpe_processor.get(), &ids);

  std::vector<std::vector<int32_t>> expected_ids(
      {{259, 1118, 234, 188, 132}, {259, 1585, 236, 161, 148, 236, 160, 191}});
  EXPECT_EQ(ids, expected_ids);
}

}  // namespace sherpa_onnx
