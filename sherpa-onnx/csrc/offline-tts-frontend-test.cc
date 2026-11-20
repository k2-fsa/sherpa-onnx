// sherpa-onnx/csrc/offline-tts-frontend-test.cc
//
// Copyright (c)  2026  linzch3

#include "sherpa-onnx/csrc/offline-tts-frontend.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

static std::unordered_map<char32_t, int32_t> MakeTokens() {
  // A tiny Kokoro-like token table
  return {{U'^', 1}, {U'$', 2}, {U' ', 3}, {U'.', 4},  {U',', 5},
          {U'w', 6}, {U'ɜ', 7}, {U'ɹ', 8}, {U'l', 9},  {U'd', 10},
          {U'ˈ', 11}};
}

static std::vector<char32_t> Repeat(const std::u32string &word, int32_t n) {
  std::u32string s;
  for (int32_t i = 0; i != n; ++i) {
    if (i) s += U' ';
    s += word;
  }
  s += U'.';
  return {s.begin(), s.end()};
}

static std::vector<int64_t> Flatten(
    const std::vector<std::vector<int64_t>> &chunks) {
  std::vector<int64_t> ans;
  for (const auto &c : chunks) ans.insert(ans.end(), c.begin(), c.end());
  return ans;
}

TEST(PiperPhonemesToIdsMatcha, LongInputIsSplitAtWordBoundary) {
  auto token2id = MakeTokens();
  // 100 x "wˈɜɹld", 7 code points per word with the space; same input as
  // https://github.com/k2-fsa/sherpa-onnx/issues/4178
  auto phonemes = Repeat(U"wˈɜɹld", 100);

  auto chunks = PiperPhonemesToIdsMatcha(token2id, phonemes, false);
  ASSERT_GT(chunks.size(), 1u);

  for (size_t i = 0; i + 1 < chunks.size(); ++i) {
    EXPECT_LE(chunks[i].size(), 402u) << "chunk " << i;
    // every chunk except the last one ends right after a space
    EXPECT_EQ(chunks[i].back(), token2id.at(U' ')) << "chunk " << i;
    // and the next one starts with the first phoneme of a word
    EXPECT_EQ(chunks[i + 1].front(), token2id.at(U'w')) << "chunk " << i + 1;
  }

  // nothing is lost or duplicated
  auto unsplit =
      PiperPhonemesToIdsMatcha(token2id, phonemes, false, /*max_token_len*/
                               100000);
  ASSERT_EQ(unsplit.size(), 1u);
  EXPECT_EQ(Flatten(chunks), unsplit[0]);
}

TEST(PiperPhonemesToIdsMatcha, LongInputWithBosEos) {
  auto token2id = MakeTokens();
  auto phonemes = Repeat(U"wˈɜɹld", 100);

  auto chunks = PiperPhonemesToIdsMatcha(token2id, phonemes, true);
  ASSERT_GT(chunks.size(), 1u);

  std::vector<int64_t> inner;
  for (size_t i = 0; i != chunks.size(); ++i) {
    const auto &c = chunks[i];
    ASSERT_GE(c.size(), 3u);
    EXPECT_EQ(c.front(), token2id.at(U'^'));
    EXPECT_EQ(c.back(), token2id.at(U'$'));
    if (i + 1 < chunks.size()) {
      EXPECT_EQ(c[c.size() - 2], token2id.at(U' ')) << "chunk " << i;
    }
    inner.insert(inner.end(), c.begin() + 1, c.end() - 1);
  }

  std::vector<int64_t> expected;
  for (auto p : phonemes) expected.push_back(token2id.at(p));
  EXPECT_EQ(inner, expected);
}

TEST(PiperPhonemesToIdsMatcha, NoBoundaryFallsBackToHardCut) {
  auto token2id = MakeTokens();
  std::vector<char32_t> phonemes(1000, U'w');

  auto chunks = PiperPhonemesToIdsMatcha(token2id, phonemes, false);
  ASSERT_EQ(chunks.size(), 3u);
  EXPECT_EQ(chunks[0].size(), 402u);
  EXPECT_EQ(chunks[1].size(), 402u);
  EXPECT_EQ(chunks[2].size(), 196u);
}

TEST(PiperPhonemesToIdsMatcha, ShortInputIsNotSplit) {
  auto token2id = MakeTokens();
  auto phonemes = Repeat(U"wˈɜɹld", 3);

  auto chunks = PiperPhonemesToIdsMatcha(token2id, phonemes, false);
  ASSERT_EQ(chunks.size(), 1u);
  EXPECT_EQ(chunks[0].size(), phonemes.size());
}

}  // namespace sherpa_onnx
