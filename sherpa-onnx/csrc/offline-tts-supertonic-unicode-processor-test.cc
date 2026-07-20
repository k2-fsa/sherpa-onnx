// sherpa-onnx/csrc/offline-tts-supertonic-unicode-processor-test.cc
//
// Copyright (c)  2026  Bogdan Radulescu

#include "sherpa-onnx/csrc/offline-tts-supertonic-unicode-processor.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

static std::vector<uint16_t> Decompose(uint32_t codepoint) {
  std::vector<uint16_t> out;
  SupertonicUnicodeProcessor::DecomposeCodepoint(codepoint, &out);
  return out;
}

TEST(SupertonicUnicodeProcessor, AsciiPassesThrough) {
  EXPECT_EQ(Decompose('a'), (std::vector<uint16_t>{'a'}));
  EXPECT_EQ(Decompose('Z'), (std::vector<uint16_t>{'Z'}));
  EXPECT_EQ(Decompose(' '), (std::vector<uint16_t>{' '}));
}

TEST(SupertonicUnicodeProcessor, Latin1Diacritics) {
  // é -> e + combining acute
  EXPECT_EQ(Decompose(0x00E9), (std::vector<uint16_t>{0x0065, 0x0301}));
  // ü -> u + combining diaeresis
  EXPECT_EQ(Decompose(0x00FC), (std::vector<uint16_t>{0x0075, 0x0308}));
  // ç -> c + combining cedilla
  EXPECT_EQ(Decompose(0x00E7), (std::vector<uint16_t>{0x0063, 0x0327}));
}

// These used to be dropped: the old hand-written table only covered
// Latin-1, so anything from Latin Extended-A/B mapped to -1 in the
// unicode indexer and was silently removed from the synthesized speech
// ("sinteză" came out as "sintez").
TEST(SupertonicUnicodeProcessor, RomanianDiacritics) {
  // ă -> a + combining breve
  EXPECT_EQ(Decompose(0x0103), (std::vector<uint16_t>{0x0061, 0x0306}));
  // ș -> s + combining comma below
  EXPECT_EQ(Decompose(0x0219), (std::vector<uint16_t>{0x0073, 0x0326}));
  // ț -> t + combining comma below
  EXPECT_EQ(Decompose(0x021B), (std::vector<uint16_t>{0x0074, 0x0326}));
  // legacy cedilla forms: ş -> s + combining cedilla
  EXPECT_EQ(Decompose(0x015F), (std::vector<uint16_t>{0x0073, 0x0327}));
  // ţ -> t + combining cedilla
  EXPECT_EQ(Decompose(0x0163), (std::vector<uint16_t>{0x0074, 0x0327}));
}

TEST(SupertonicUnicodeProcessor, OtherLatinExtended) {
  // Czech: č -> c + combining caron
  EXPECT_EQ(Decompose(0x010D), (std::vector<uint16_t>{0x0063, 0x030C}));
  // Polish: ą -> a + combining ogonek
  EXPECT_EQ(Decompose(0x0105), (std::vector<uint16_t>{0x0061, 0x0328}));
  // Hungarian: ő -> o + combining double acute
  EXPECT_EQ(Decompose(0x0151), (std::vector<uint16_t>{0x006F, 0x030B}));
  // Turkish: ğ -> g + combining breve
  EXPECT_EQ(Decompose(0x011F), (std::vector<uint16_t>{0x0067, 0x0306}));
  // Vietnamese: ế -> e + circumflex + acute
  EXPECT_EQ(Decompose(0x1EBF), (std::vector<uint16_t>{0x0065, 0x0302, 0x0301}));
}

TEST(SupertonicUnicodeProcessor, CharactersWithoutDecomposition) {
  // ł and ı have no canonical decomposition and are in the indexer as-is
  EXPECT_EQ(Decompose(0x0142), (std::vector<uint16_t>{0x0142}));
  EXPECT_EQ(Decompose(0x0131), (std::vector<uint16_t>{0x0131}));
}

TEST(SupertonicUnicodeProcessor, HangulIsAlgorithmic) {
  // 한 -> ᄒ + ᅡ + ᆫ
  EXPECT_EQ(Decompose(0xD55C), (std::vector<uint16_t>{0x1112, 0x1161, 0x11AB}));
  // 가 -> ᄀ + ᅡ (no trailing consonant)
  EXPECT_EQ(Decompose(0xAC00), (std::vector<uint16_t>{0x1100, 0x1161}));
}

TEST(SupertonicUnicodeProcessor, CompatibilityDecompositions) {
  // NFKD also handles compatibility forms, like the reference
  // implementation (unicodedata.normalize("NFKD", text)).
  // Fullwidth Ａ -> A
  EXPECT_EQ(Decompose(0xFF21), (std::vector<uint16_t>{0x0041}));
  // ﬁ ligature -> f + i
  EXPECT_EQ(Decompose(0xFB01), (std::vector<uint16_t>{0x0066, 0x0069}));
}

TEST(SupertonicUnicodeProcessor, NonBmpIsDropped) {
  EXPECT_TRUE(Decompose(0x1F600).empty());
}

}  // namespace sherpa_onnx
