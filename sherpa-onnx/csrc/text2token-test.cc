// sherpa-onnx/csrc/text2token-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/file-utils.h"
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

  if (!FileExists(tokens)) {
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
  if (!FileExists(tokens) || !FileExists(bpe)) {
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
  if (!FileExists(tokens) || !FileExists(bpe)) {
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
  if (!FileExists(tokens) || !FileExists(bpe)) {
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

TEST(TEXT2TOKEN, TEST_bbpe_non_cjk) {
  // bbpe models are trained on byte_encode(tokenize_by_CJK_char(text)) in
  // icefall, so "你好HELLO" is seen as "你 好 HELLO": one word per CJK
  // character, while "HELLO" stays one word. A tiny vocabulary in the
  // byte-mapped form is enough to check this. After mapping each byte,
  // 你 is "ƋţŅ" and 好 is "ƌŋţ".
  std::string vocab =
      "<unk> 0\n"
      "▁ -10\n"
      "H -10\n"
      "E -10\n"
      "L -10\n"
      "O -10\n"
      "▁HELLO -1\n"
      "▁ƋţŅ -1\n"
      "▁ƌŋţ -1\n";
  std::istringstream vocab_is(vocab);
  auto bpe_processor =
      std::make_unique<ssentencepiece::Ssentencepiece>(vocab_is);

  std::string tokens =
      "<unk> 0\n"
      "▁ 1\n"
      "H 2\n"
      "E 3\n"
      "L 4\n"
      "O 5\n"
      "▁HELLO 6\n"
      "▁ƋţŅ 7\n"
      "▁ƌŋţ 8\n";
  auto sym_table = SymbolTable(tokens, false);

  std::string text = "HELLO :2.0\n你好HELLO\n你好";
  std::istringstream iss(text);

  std::vector<std::vector<int32_t>> ids;
  std::vector<float> scores;

  auto r = EncodeHotwords(iss, "bbpe", sym_table, bpe_processor.get(), &ids,
                          &scores);
  EXPECT_TRUE(r);

  std::vector<std::vector<int32_t>> expected_ids({{6}, {7, 8, 6}, {7, 8}});
  EXPECT_EQ(ids, expected_ids);

  std::vector<float> expected_scores({2.0, 0, 0});
  EXPECT_EQ(scores, expected_scores);
}

}  // namespace sherpa_onnx
