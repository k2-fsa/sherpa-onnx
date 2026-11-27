// sherpa-onnx/csrc/offline-tts-frontend-test.cc
//
// Copyright (c)  2026  kaveen

#include "sherpa-onnx/csrc/offline-tts-frontend.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

// espeak-ng outputs "ç" in "ich" as "c" + U+0327 (combining cedilla)
static const std::vector<char32_t> kIch = {0x026a, U'c', 0x0327};

TEST(OfflineTtsFrontend, CombiningCedillaIsMerged) {
  std::unordered_map<char32_t, int32_t> token2id = {
      {0x026a, 1}, {U'c', 2}, {0x00e7, 3}};

  auto ids = PiperPhonemesToIdsMatcha(token2id, kIch, false);
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], (std::vector<int64_t>{1, 3}));
}

TEST(OfflineTtsFrontend, CombiningCedillaIsKeptIfItIsAToken) {
  std::unordered_map<char32_t, int32_t> token2id = {
      {0x026a, 1}, {U'c', 2}, {0x00e7, 3}, {0x0327, 4}};

  auto ids = PiperPhonemesToIdsMatcha(token2id, kIch, false);
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], (std::vector<int64_t>{1, 2, 4}));
}

TEST(OfflineTtsFrontend, NoChangeWithoutCedillaToken) {
  std::unordered_map<char32_t, int32_t> token2id = {{0x026a, 1}, {U'c', 2}};

  auto ids = PiperPhonemesToIdsMatcha(token2id, kIch, false);
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], (std::vector<int64_t>{1, 2}));
}

}  // namespace sherpa_onnx
