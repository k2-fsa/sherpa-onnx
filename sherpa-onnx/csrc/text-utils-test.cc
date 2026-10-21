// sherpa-onnx/csrc/text-utils-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/text-utils.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(QuoteJsonString, EmptyAndOrdinaryText) {
  EXPECT_EQ(QuoteJsonString(""), "\"\"");
  EXPECT_EQ(QuoteJsonString("plain / text"), "\"plain / text\"");
  EXPECT_EQ(QuoteJsonString("你好世界 🌍"), "\"你好世界 🌍\"");
  EXPECT_EQ(QuoteJsonString("\x7f"), "\"\x7f\"");
}

TEST(QuoteJsonString, QuotesAndBackslashes) {
  EXPECT_EQ(QuoteJsonString("\"quoted\" \\ path"),
            "\"\\\"quoted\\\" \\\\ path\"");
}

TEST(QuoteJsonString, AllControlCharacters) {
  std::string text;
  for (int32_t c = 0; c < 0x20; ++c) {
    text.push_back(static_cast<char>(c));
  }
  EXPECT_EQ(QuoteJsonString(text),
            "\"\\u0000\\u0001\\u0002\\u0003\\u0004\\u0005\\u0006\\u0007"
            "\\b\\t\\n\\u000b\\f\\r\\u000e\\u000f"
            "\\u0010\\u0011\\u0012\\u0013\\u0014\\u0015\\u0016\\u0017"
            "\\u0018\\u0019\\u001a\\u001b\\u001c\\u001d\\u001e\\u001f\"");
}

TEST(ToLowerCase, WideString) {
  std::string text =
      "Hallo! Übeltäter übergibt Ärzten öfters äußerst ätzende Öle 3€";
  auto t = ToLowerCase(text);
  std::cout << text << "\n";
  std::cout << t << "\n";
}

TEST(RemoveInvalidUtf8Sequences, Case1) {
  std::vector<uint8_t> v = {
      0xe4, 0xbb, 0x8a,                                  // 今
      0xe5, 0xa4, 0xa9,                                  // 天
      'i',  's',  ' ',  'M', 'o', 'd', 'a', 'y',  ',',   // is Monday,
      ' ',  'w',  'i',  'e', ' ', 'h', 'e', 'i',  0xc3,  // wie heißen Size
      0x9f, 'e',  'n',  ' ', 'S', 'i', 'e', 0xf0, 0x9d, 0x84, 0x81};

  std::vector<uint8_t> v0 = v;
  v0[1] = 0xc0;  // make the first 3 bytes an invalid utf8 character
  std::string s0{v0.begin(), v0.end()};
  EXPECT_EQ(s0.size(), v0.size());

  auto s = RemoveInvalidUtf8Sequences(s0);  // should remove 今

  v0 = v;
  // v0[23] == 0xc3
  // v0[24] == 0x9f

  v0[23] = 0xc1;

  s0 = {v0.begin(), v0.end()};
  s = RemoveInvalidUtf8Sequences(s0);  // should remove ß

  EXPECT_EQ(s.size() + 2, v.size());

  v0 = v;
  // v0[31] = 0xf0;
  // v0[32] = 0x9d;
  // v0[33] = 0x84;
  // v0[34] = 0x81;
  v0[31] = 0xf5;

  s0 = {v0.begin(), v0.end()};
  s = RemoveInvalidUtf8Sequences(s0);

  EXPECT_EQ(s.size() + 4, v.size());
}

// Tests for sanitizeUtf8
TEST(RemoveInvalidUtf8Sequences, ValidUtf8StringPassesUnchanged) {
  std::string input = "Valid UTF-8 🌍";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), input);
}

TEST(RemoveInvalidUtf8Sequences, SingleInvalidByteReplaced) {
  std::string input = "Invalid \xFF UTF-8";
  std::string expected = "Invalid  UTF-8";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, TruncatedUtf8SequenceReplaced) {
  std::string input = "Broken \xE2\x82";  // Incomplete UTF-8 sequence
  std::string expected = "Broken ";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, MultipleInvalidBytes) {
  std::string input = "Test \xC0\xC0\xF8\xA0";  // Multiple invalid sequences
  std::string expected = "Test ";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, BreakingCase_SpaceFollowedByInvalidByte) {
  std::string input = "\x20\xC4";  // Space followed by an invalid byte
  std::string expected = " ";      // 0xC4 removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, ValidUtf8WithEdgeCaseCharacters) {
  std::string input = "Edge 🏆💯";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), input);
}

TEST(RemoveInvalidUtf8Sequences, MixedValidAndInvalidBytes) {
  std::string input = "Mix \xE2\x82\xAC \xF0\x9F\x98\x81 \xFF";
  std::string expected = "Mix € 😁 ";  // Invalid bytes removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, SpaceFollowedByInvalidByte) {
  std::string input = "\x20\xC4";  // Space (0x20) followed by invalid (0xC4)
  std::string expected = " ";      // Space remains, 0xC4 is removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, RemoveTruncatedC4) {
  std::string input = "Hello \xc4 world";  // Invalid `0xC4`
  std::string expected = "Hello  world";   // `0xC4` should be removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, SpaceFollowedByInvalidByte_Breaking) {
  std::string input = "\x20\xc4";  // Space followed by invalid `0xc4`
  std::string expected = " ";      // `0xc4` should be removed, space remains
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, DebugSpaceFollowedByInvalidByte) {
  std::string input = "\x20\xc4";  // Space followed by invalid `0xc4`
  std::string output = RemoveInvalidUtf8Sequences(input);

  std::cout << "Processed string: ";
  for (unsigned char c : output) {
    printf("\\x%02x ", c);
  }
  std::cout << std::endl;

  EXPECT_EQ(output, " ");  // Expect `0xc4` to be removed, leaving only space
}

TEST(SplitUtf8, TruncatedSequenceAtEndIsSkipped) {
  // "今天天气很好" without the last byte of "好"
  std::string text(
      "\xe4\xbb\x8a\xe5\xa4\xa9\xe5\xa4\xa9"
      "\xe6\xb0\x94\xe5\xbe\x88\xe5\xa5",
      17);
  std::vector<std::string> expected = {"今", "天", "天", "气", "很"};
  EXPECT_EQ(SplitUtf8(text), expected);

  // The first byte of a 4-byte sequence, and nothing after it
  std::string text2("0123456789abcdef\xf0", 17);
  EXPECT_EQ(SplitUtf8(text2), std::vector<std::string>{"0123456789abcdef"});
}

TEST(RemoveInvalidUtf8Sequences, ByteLevelBpeTokenFragment) {
  // Whisper byte-level-BPE tokens can split a multi-byte UTF-8 character
  // across two tokens. Each fragment alone is invalid UTF-8 and is
  // stripped; the character only survives when the fragments are joined
  // first. Recognizer implementations must therefore concatenate tokens
  // before calling this function (see the whisper Convert()).
  std::string frag1 = "\xd7";  // leading byte of a Hebrew letter
  std::string frag2 = "\xa8";  // continuation byte (frag1+frag2 is a letter)
  EXPECT_EQ(RemoveInvalidUtf8Sequences(frag1), "");
  EXPECT_EQ(RemoveInvalidUtf8Sequences(frag2), "");
  EXPECT_EQ(RemoveInvalidUtf8Sequences(frag1 + frag2), "\xd7\xa8");
}

TEST(SplitByAllPunctuation, English) {
  auto result = SplitByAllPunctuation("Hello, world. How are you?");
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0], "Hello,");
  EXPECT_EQ(result[1], "world.");
  EXPECT_EQ(result[2], "How are you?");
}

TEST(SplitByAllPunctuation, Chinese) {
  auto result = SplitByAllPunctuation("你好，世界。你好吗？");
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0], "你好，");
  EXPECT_EQ(result[1], "世界。");
  EXPECT_EQ(result[2], "你好吗？");
}

TEST(SplitByAllPunctuation, MixedPunctuation) {
  auto result = SplitByAllPunctuation("a;b:c,d");
  ASSERT_EQ(result.size(), 4u);
  EXPECT_EQ(result[0], "a;");
  EXPECT_EQ(result[1], "b:");
  EXPECT_EQ(result[2], "c,");
  EXPECT_EQ(result[3], "d");
}

TEST(SplitByAllPunctuation, NoPunctuation) {
  auto result = SplitByAllPunctuation("hello world");
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], "hello world");
}

TEST(SplitByAllPunctuation, Empty) {
  auto result = SplitByAllPunctuation("");
  EXPECT_EQ(result.size(), 0u);
}

TEST(CountWords, English) {
  EXPECT_EQ(CountWords("hello world"), 2);
  EXPECT_EQ(CountWords("one two three four five"), 5);
  EXPECT_EQ(CountWords(""), 0);
}

TEST(CountWords, Chinese) {
  // Each CJK character is one word
  EXPECT_EQ(CountWords("你好世界"), 4);
  EXPECT_EQ(CountWords("当夜幕降临"), 5);
}

TEST(CountWords, Mixed) {
  // Each CJK char = 1 word, each English word = 1 word
  // "hello"(1) "你"(2) "好"(3) "world"(4) = 4
  EXPECT_EQ(CountWords("hello 你好 world"), 4);
}

TEST(CountWords, Punctuation) {
  // Sentence-ending punctuation counts as a word
  EXPECT_EQ(CountWords("hello."), 2);
  // Commas/semicolons are word boundaries, not words
  EXPECT_EQ(CountWords("a,b,c"), 3);
}

TEST(MergeShortSentencesByWords, Basic) {
  std::vector<std::string> sentences = {"hello", "world"};
  auto result = MergeShortSentencesByWords(sentences, 5);
  // "hello" has 1 word < 5, merged with "world" → "helloworld"
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], "helloworld");
}

TEST(MergeShortSentencesByWords, LongEnough) {
  std::vector<std::string> sentences = {"one two three four five",
                                        "six seven eight nine ten"};
  auto result = MergeShortSentencesByWords(sentences, 5);
  // Both have >= 5 words, not merged
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], "one two three four five");
  EXPECT_EQ(result[1], "six seven eight nine ten");
}

TEST(MergeShortSentencesByWords, PunctuationOnlyMerged) {
  std::vector<std::string> sentences = {"hello world", "."};
  auto result = MergeShortSentencesByWords(sentences, 5);
  // "." is all punctuation, merged into previous
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], "hello world.");
}

TEST(SplitLongSentenceByWords, Short) {
  auto result = SplitLongSentenceByWords("hello world", 20);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], "hello world");
}

TEST(SplitLongSentenceByWords, SplitAtSpace) {
  // "a b c d e" has 5 words, max=3 → split at space boundary
  auto result = SplitLongSentenceByWords("a b c d e", 3);
  ASSERT_EQ(result.size(), 2u);
  // The space at the split boundary is consumed
  EXPECT_EQ(result[0], "a b c");
  EXPECT_EQ(result[1], "d e");
}

TEST(SplitLongSentenceByWords, Empty) {
  auto result = SplitLongSentenceByWords("", 20);
  EXPECT_EQ(result.size(), 0u);
}

TEST(SplitLongSentenceByWords, NoHangOnLeadingSpaceAfterSplit) {
  // Regression: the first chunk ends on a space, so the second chunk starts
  // with that space. If it then needs another split and has no internal
  // boundary, the backward search finds the leading space and would set
  // split_pos == start, never advancing. Must terminate and make progress.
  auto result = SplitLongSentenceByWords("ab 你好世界你好世界你好世界你好世界你好世界你好", 20);
  EXPECT_GT(result.size(), 1u);

  // The pieces must still cover the input without dropping content.
  std::string joined;
  for (const auto &p : result) joined += p;
  EXPECT_EQ(joined, "ab你好世界你好世界你好世界你好世界你好世界你好");
}

TEST(SplitLongSentenceByWords, AdvancesOnSpaceOnlySeparator) {
  // Two words separated by a single space, with max_words=1: each split lands
  // on the space, so without skipping leading spaces the loop would spin.
  auto result = SplitLongSentenceByWords("aa bb", 1);
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], "aa");
  EXPECT_EQ(result[1], "bb");
}

TEST(ParseLexiconFile, NewFormat) {
  std::istringstream is(
      "# comment\n"
      "hello || h ə l oʊ\n"
      "world || w ɜː l d\n");
  int32_t max_len = 0;
  auto entries = ParseLexiconFile(is, &max_len);
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].key, "hello");
  // "oʊ" ends with ʊ (not digit) → split into "o" + "ʊ"
  ASSERT_EQ(entries[0].phonemes.size(), 5u);
  EXPECT_EQ(entries[0].phonemes[0], "h");
  EXPECT_EQ(entries[0].phonemes[1], "ə");
  EXPECT_EQ(entries[0].phonemes[2], "l");
  EXPECT_EQ(entries[0].phonemes[3], "o");
  EXPECT_EQ(entries[0].phonemes[4], "ʊ");
  EXPECT_EQ(entries[1].key, "world");
  EXPECT_EQ(max_len, 5);
}

TEST(ParseLexiconFile, OldFormat) {
  std::istringstream is(
      "# comment\n"
      "zhong1 z h o ng 1\n"
      "guo2 g u o 2\n");
  int32_t max_len = 0;
  auto entries = ParseLexiconFile(is, &max_len);
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].key, "zhong1");
  // "ng" ends with 'g' (letter) → kept whole
  ASSERT_EQ(entries[0].phonemes.size(), 5u);
  EXPECT_EQ(entries[0].phonemes[0], "z");
  EXPECT_EQ(entries[0].phonemes[1], "h");
  EXPECT_EQ(entries[0].phonemes[2], "o");
  EXPECT_EQ(entries[0].phonemes[3], "ng");
  EXPECT_EQ(entries[0].phonemes[4], "1");
  EXPECT_EQ(entries[1].key, "guo2");
  EXPECT_EQ(max_len, 6);
}

TEST(ParseLexiconFile, MultiWord) {
  std::istringstream is("New York || n j uː ˈ j ɔː k\n");
  int32_t max_len = 0;
  auto entries = ParseLexiconFile(is, &max_len);
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].key, "new york");
  // "uː" ends with ː (not digit) → split into "u" + "ː"
  ASSERT_EQ(entries[0].phonemes.size(), 9u);
  EXPECT_EQ(entries[0].phonemes[0], "n");
  EXPECT_EQ(entries[0].phonemes[1], "j");
  EXPECT_EQ(entries[0].phonemes[2], "u");
  EXPECT_EQ(entries[0].phonemes[3], "ː");
  EXPECT_EQ(max_len, 8);
}

TEST(ParseLexiconFile, PinyinKeptWhole) {
  // Pinyin tokens like "ao3" end with a digit → NOT split
  std::istringstream is("好 h ao3\n");
  int32_t max_len = 0;
  auto entries = ParseLexiconFile(is, &max_len);
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].key, "好");
  ASSERT_EQ(entries[0].phonemes.size(), 2u);
  EXPECT_EQ(entries[0].phonemes[0], "h");
  EXPECT_EQ(entries[0].phonemes[1], "ao3");  // kept whole
}

TEST(ParseLexiconFile, MultiByteLastCodepoint) {
  // "ɑː" is ɑ (U+0251, 2 bytes) + ː (U+02D0, 2 bytes)
  // Last codepoint ː is not alphanumeric → split into ɑ + ː
  // Verify we check the codepoint, not the byte.
  std::istringstream is("foo || ɑː l\n");
  int32_t max_len = 0;
  auto entries = ParseLexiconFile(is, &max_len);
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].key, "foo");
  ASSERT_EQ(entries[0].phonemes.size(), 3u);
  EXPECT_EQ(entries[0].phonemes[0], "ɑ");
  EXPECT_EQ(entries[0].phonemes[1], "ː");
  EXPECT_EQ(entries[0].phonemes[2], "l");
}

TEST(ParseLexiconFile, EmptyAndComments) {
  std::istringstream is("# comment\n\n  \n# another\n");
  int32_t max_len = 0;
  auto entries = ParseLexiconFile(is, &max_len);
  EXPECT_EQ(entries.size(), 0u);
  EXPECT_EQ(max_len, 0);
}

TEST(ParseLexiconFile, NullMaxLen) {
  std::istringstream is("hello || h ə l oʊ\n");
  auto entries = ParseLexiconFile(is, nullptr);
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].key, "hello");
}

TEST(ParseLexiconFile, MultiCodepointPhonemes) {
  // "dˈ" ends with ˈ (not digit) → split into "d" + "ˈ"
  // "ao3" ends with 3 (digit) → kept as "ao3"
  std::istringstream is("today || t ə dˈ ao3 ɪ\n");
  int32_t max_len = 0;
  auto entries = ParseLexiconFile(is, &max_len);
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].key, "today");
  ASSERT_EQ(entries[0].phonemes.size(), 6u);
  EXPECT_EQ(entries[0].phonemes[0], "t");
  EXPECT_EQ(entries[0].phonemes[1], "ə");
  EXPECT_EQ(entries[0].phonemes[2], "d");
  EXPECT_EQ(entries[0].phonemes[3], "ˈ");
  EXPECT_EQ(entries[0].phonemes[4], "ao3");  // ends with digit, kept whole
  EXPECT_EQ(entries[0].phonemes[5], "ɪ");
}

TEST(Utf8ToUtf16, Bmp) {
  // a, U+00E9, U+20AC
  std::u16string expected = {u'a', 0x00E9, 0x20AC};
  EXPECT_EQ(Utf8ToUtf16("a\xc3\xa9\xe2\x82\xac"), expected);
}

TEST(Utf8ToUtf16, SurrogatePairs) {
  // U+1F600 and U+20000 become surrogate pairs; the text after them is kept
  std::u16string expected = {u'a', 0xD83D, 0xDE00, u'b', 0xD840, 0xDC00, u'c'};
  EXPECT_EQ(Utf8ToUtf16("a\xf0\x9f\x98\x80"
                        "b\xf0\xa0\x80\x80"
                        "c"),
            expected);

  std::u16string max = {0xDBFF, 0xDFFF};
  EXPECT_EQ(Utf8ToUtf16("\xf4\x8f\xbf\xbf"), max);  // U+10FFFF
}

TEST(Utf8ToUtf16, InvalidBytes) {
  // an encoded surrogate (modified UTF-8, ed a0 bd) is not valid UTF-8
  std::u16string expected = {0xFFFD, 0xFFFD, 0xFFFD, u'a'};
  EXPECT_EQ(Utf8ToUtf16("\xed\xa0\xbd"
                        "a"),
            expected);
}

TEST(Utf8ToUtf16, EmbeddedNull) {
  // the text after U+0000 is kept (modified UTF-8 would use c0 80)
  std::u16string expected = {u'a', 0, u'b'};
  EXPECT_EQ(Utf8ToUtf16(std::string("a\0b", 3)), expected);
}

TEST(Utf8ToUtf16, Empty) { EXPECT_EQ(Utf8ToUtf16(""), u""); }

TEST(Utf16ToUtf8, Bmp) {
  // A, é, €, →
  EXPECT_EQ(Utf16ToUtf8(u"A\u00e9\u20ac\u2192"),
            "A\xc3\xa9\xe2\x82\xac\xe2\x86\x92");
}

TEST(Utf16ToUtf8, SurrogatePairs) {
  // U+1F600 and U+20000 must be 4-byte sequences, not modified UTF-8
  // (ed a0 bd ed b8 80 for U+1F600)
  EXPECT_EQ(Utf16ToUtf8(u"\U0001F600"), "\xf0\x9f\x98\x80");
  EXPECT_EQ(Utf16ToUtf8(u"a\U00020000b"),
            "a\xf0\xa0\x80\x80"
            "b");
  EXPECT_EQ(Utf16ToUtf8(u"\U0010FFFF"), "\xf4\x8f\xbf\xbf");
}

TEST(Utf16ToUtf8, UnpairedSurrogates) {
  std::u16string high_at_end = {u'a', static_cast<char16_t>(0xD83D)};
  EXPECT_EQ(Utf16ToUtf8(high_at_end), "a\xef\xbf\xbd");

  std::u16string high_then_bmp = {static_cast<char16_t>(0xD83D), u'a'};
  EXPECT_EQ(Utf16ToUtf8(high_then_bmp),
            "\xef\xbf\xbd"
            "a");

  std::u16string lone_low = {static_cast<char16_t>(0xDE00), u'a'};
  EXPECT_EQ(Utf16ToUtf8(lone_low),
            "\xef\xbf\xbd"
            "a");

  // low before high is not a pair
  std::u16string reversed = {static_cast<char16_t>(0xDE00),
                             static_cast<char16_t>(0xD83D)};
  EXPECT_EQ(Utf16ToUtf8(reversed), "\xef\xbf\xbd\xef\xbf\xbd");
}

TEST(Utf16ToUtf8, EmbeddedNull) {
  // standard UTF-8 uses a 0 byte, not c0 80 as in modified UTF-8
  std::u16string s = {u'a', u'\0', u'b'};
  EXPECT_EQ(Utf16ToUtf8(s), std::string("a\0b", 3));
}

TEST(Utf16ToUtf8, Empty) { EXPECT_EQ(Utf16ToUtf8(u""), ""); }

}  // namespace sherpa_onnx
