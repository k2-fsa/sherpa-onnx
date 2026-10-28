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


// Each test corrupts the lead byte of one multi-byte character in the same
// base string and checks that exactly that character is dropped. Expectations
// are literal results, so a failure shows the text rather than a size.
static const char *kMixedScriptText = "\xe4\xbb\x8a\xe5\xa4\xa9is Monday, wie hei\xc3\x9f" "en Sie\xf0\x9d\x86\x81";

TEST(RemoveInvalidUtf8Sequences, CorruptingThreeByteCharRemovesIt) {
  // "今" occupies bytes 0..2.
  std::string input = kMixedScriptText;
  input[0] = static_cast<char>(0xC0);
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), "\xe5\xa4\xa9is Monday, wie hei\xc3\x9f" "en Sie\xf0\x9d\x86\x81");
}

TEST(RemoveInvalidUtf8Sequences, CorruptingTwoByteCharRemovesIt) {
  // "ß" occupies bytes 24..25.
  std::string input = kMixedScriptText;
  input[24] = static_cast<char>(0xC1);
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), "\xe4\xbb\x8a\xe5\xa4\xa9is Monday, wie heien Sie\xf0\x9d\x86\x81");
}

TEST(RemoveInvalidUtf8Sequences, CorruptingFourByteCharRemovesIt) {
  // The musical symbol occupies bytes 32..35; 0xF5 is not a valid lead byte.
  std::string input = kMixedScriptText;
  input[32] = static_cast<char>(0xF5);
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), "\xe4\xbb\x8a\xe5\xa4\xa9is Monday, wie hei\xc3\x9f" "en Sie");
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

TEST(RemoveInvalidUtf8Sequences, ValidUtf8WithEdgeCaseCharacters) {
  std::string input = "Edge 🏆💯";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), input);
}

TEST(RemoveInvalidUtf8Sequences, MixedValidAndInvalidBytes) {
  std::string input = "Mix \xE2\x82\xAC \xF0\x9F\x98\x81 \xFF";
  std::string expected = "Mix € 😁 ";  // Invalid bytes removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, TruncatedLeadByteIsDropped) {
  // 0xC4 is a 2-byte lead byte with no continuation: the whole byte goes.
  // The surrounding text is untouched, including the space before it.
  EXPECT_EQ(RemoveInvalidUtf8Sequences(std::string("\x20\xc4")), " ");
  EXPECT_EQ(RemoveInvalidUtf8Sequences("Hello \xc4 world"), "Hello  world");
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
  // max_len counts words, not codepoints (#4020). Both keys are one word.
  EXPECT_EQ(max_len, 1);
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
  // max_len counts words, not codepoints (#4020).
  EXPECT_EQ(max_len, 1);
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
  // max_len counts lookup tokens (#4020): "new york" is two runs plus the
  // space token, and the space-preserving frontends need all three.
  EXPECT_EQ(max_len, 3);
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

// ---------------------------------------------------------------------
// #4020: lexicon matching is case-insensitive and form-insensitive.
// ---------------------------------------------------------------------

TEST(ToSimpleLower, Ascii) {
  EXPECT_EQ(ToSimpleLower(U'A'), U'a');
  EXPECT_EQ(ToSimpleLower(U'z'), U'z');
  EXPECT_EQ(ToSimpleLower(U'0'), U'0');
}

TEST(ToSimpleLower, LatinExtended) {
  // The original hand-maintained switch could not do these: std::towlower()
  // is a no-op for non-ASCII in the "C" locale.
  EXPECT_EQ(ToSimpleLower(U'Ā'), U'ā');  // U+0100 -> U+0101
  EXPECT_EQ(ToSimpleLower(U'Ī'), U'ī');  // U+012A -> U+012B
  EXPECT_EQ(ToSimpleLower(U'Ñ'), U'ñ');  // U+00D1 -> U+00F1
  EXPECT_EQ(ToSimpleLower(U'Ü'), U'ü');
}

TEST(ToSimpleLower, OtherScripts) {
  EXPECT_EQ(ToSimpleLower(U'Α'), U'α');  // Greek
  EXPECT_EQ(ToSimpleLower(U'Б'), U'б');  // Cyrillic
  EXPECT_EQ(ToSimpleLower(U'Ա'), U'ա');  // Armenian
}

TEST(ToSimpleLower, UncasedIsUnchanged) {
  EXPECT_EQ(ToSimpleLower(U'中'), U'中');  // CJK
  EXPECT_EQ(ToSimpleLower(U'ก'), U'ก');  // Thai
}

TEST(IsLexiconWordChar, LettersAndMarks) {
  EXPECT_TRUE(IsLexiconWordChar(U'a'));
  EXPECT_TRUE(IsLexiconWordChar(U'Z'));
  EXPECT_TRUE(IsLexiconWordChar(U'ā'));   // precomposed
  EXPECT_TRUE(IsLexiconWordChar(U'́'));  // combining mark
  EXPECT_TRUE(IsLexiconWordChar(U'中'));
}

TEST(IsLexiconWordChar, PunctuationIsNotAWordChar) {
  EXPECT_FALSE(IsLexiconWordChar(U' '));
  EXPECT_FALSE(IsLexiconWordChar(U','));
  EXPECT_FALSE(IsLexiconWordChar(U'“'));  // U+201C
  EXPECT_FALSE(IsLexiconWordChar(U'—'));  // U+2014
  EXPECT_FALSE(IsLexiconWordChar(U'('));
}

TEST(NormalizeNfc, ComposesBasePlusMark) {
  // "a" + COMBINING MACRON must become U+0101 so it matches a lexicon entry.
  std::u32string s = {U'a', U'̄'};
  EXPECT_TRUE(NormalizeNfc(&s));
  EXPECT_EQ(s, std::u32string(1, U'ā'));
}

TEST(NormalizeNfc, ComposesCapitalBase) {
  // U+0041 + COMBINING MACRON -> U+0100. Must compose before lowercasing,
  // otherwise ToSimpleLower(U+0100) never sees it.
  std::u32string s = {U'A', U'̄'};
  NormalizeNfc(&s);
  EXPECT_EQ(s, std::u32string(1, U'Ā'));
  EXPECT_EQ(ToSimpleLower(s[0]), U'ā');
}

TEST(NormalizeNfc, AlreadyComposedIsUnchanged) {
  std::u32string s = {U'ā'};
  EXPECT_FALSE(NormalizeNfc(&s));
  EXPECT_EQ(s, std::u32string(1, U'ā'));
}

TEST(NormalizeNfc, Utf8Overload) {
  // a + U+0301 in UTF-8
  EXPECT_EQ(NormalizeNfc(std::string("a\xcc\x81")), std::string("\xc3\xa1"));
}

TEST(ToLowerCase, NonAsciiIsLowercased) {
  // Regression: ToLowerCase used std::tolower per byte, which is a no-op for
  // UTF-8 multibyte sequences.
  EXPECT_EQ(ToLowerCase(std::string("\xc4\x80")), std::string("\xc4\x81"));  // Ā -> ā
  EXPECT_EQ(ToLowerCase("ABC"), "abc");
}

TEST(NormalizeLexiconKey, ComposesThenLowercases) {
  // "A" + COMBINING MACRON must end up as the same key as U+0101, whichever
  // form the lexicon and the query use.
  EXPECT_EQ(NormalizeLexiconKey(std::string("A\xcc\x84")),
            NormalizeLexiconKey(std::string("\xc4\x81")));
  EXPECT_EQ(NormalizeLexiconKey(std::string("A\xcc\x84")), std::string("\xc4\x81"));
}

TEST(CountLexiconPhraseWords, LatinRuns) {
  EXPECT_EQ(CountLexiconPhraseWords("hello"), 1);
  EXPECT_EQ(CountLexiconPhraseWords("hello world"), 3);   // run, space, run
  EXPECT_EQ(CountLexiconPhraseWords("new york city"), 5);
  EXPECT_EQ(CountLexiconPhraseWords(""), 0);
}

TEST(CountLexiconPhraseWords, CjkCountsOnePerCharacter) {
  // The CJK frontends tokenize one character at a time, so a two-character
  // key must report 2 or the longest-match loop never tries it.
  EXPECT_EQ(CountLexiconPhraseWords("中"), 1);
  EXPECT_EQ(CountLexiconPhraseWords("中国"), 2);
  EXPECT_EQ(CountLexiconPhraseWords("中国北京"), 4);
}

TEST(CountLexiconPhraseWords, SpacesAndPunctuationCountAsTokens) {
  // Kokoro, Kitten and Matcha keep spaces and punctuation as tokens and those
  // consume `len` in the longest-match loop. "new   york" therefore needs a
  // bound of 5 there, and an under-estimate loses the phrase entirely.
  EXPECT_EQ(CountLexiconPhraseWords("new   york"), 5);
  EXPECT_EQ(CountLexiconPhraseWords("hello, world"), 4);  // run , space run
  EXPECT_EQ(CountLexiconPhraseWords("a.b"), 3);
}

TEST(CountLexiconPhraseWords, MixedText) {
  // A non-CJK run is one token, each CJK character is one token.
  EXPECT_EQ(CountLexiconPhraseWords("hello中国"), 3);
  EXPECT_EQ(CountLexiconPhraseWords("zhong1"), 1);
  EXPECT_EQ(CountLexiconPhraseWords("zhong1 guo2"), 3);
}

TEST(CountLexiconPhraseWords, SingleWordLexiconsGetABoundOfOne) {
  // The case that motivated tightening the bound at all: a lexicon of plain
  // words must not make every position try a 40-token phrase.
  EXPECT_EQ(CountLexiconPhraseWords("rūpaṁ"), 1);
  EXPECT_EQ(CountLexiconPhraseWords("saṅkhārā"), 1);
}

// ---------------------------------------------------------------------
// #4020 follow-up review findings.
// ---------------------------------------------------------------------

TEST(IsLexiconWordChar, DigitsAreWordChars) {
  // Regression: pinyin keys like "zhong1-guo2" must keep their tone digits
  // when a punctuation-bearing token is split before lookup.
  EXPECT_TRUE(IsLexiconWordChar(U'0'));
  EXPECT_TRUE(IsLexiconWordChar(U'1'));
  EXPECT_TRUE(IsLexiconWordChar(U'9'));
  EXPECT_TRUE(IsLexiconWordChar(U'٣'));  // Arabic-Indic digit, category Nd
  EXPECT_FALSE(IsLexiconWordChar(U','));
  EXPECT_FALSE(IsLexiconWordChar(U'—'));
}

TEST(NormalizeNfc, StarterPlusStarterComposes) {
  // UAX #15 D117: a starter may compose with the starter directly before it.
  // These are NOT mark attachments -- ccc(U+09BE) == 0.
  {
    // Bengali U+09CB = U+09C7 + U+09BE
    std::u32string s = {0x9C7, 0x9BE};
    NormalizeNfc(&s);
    EXPECT_EQ(s, std::u32string(1, static_cast<char32_t>(0x9CB)));
  }
  {
    // Tamil U+0BCA = U+0BC6 + U+0BBE
    std::u32string s = {0xBC6, 0xBBE};
    NormalizeNfc(&s);
    EXPECT_EQ(s, std::u32string(1, static_cast<char32_t>(0xBCA)));
  }
  {
    // Already-composed input must be left alone (it decomposes and must
    // recompose back).
    std::u32string s = {static_cast<char32_t>(0x9CB)};
    NormalizeNfc(&s);
    EXPECT_EQ(s, std::u32string(1, static_cast<char32_t>(0x9CB)));
  }
}

TEST(NormalizeNfc, HangulComposesAlgorithmically) {
  // L + V -> LV
  {
    std::u32string s = {0x1100, 0x1161};  // choseong kiyeok + jungseong a
    NormalizeNfc(&s);
    EXPECT_EQ(s, std::u32string(1, static_cast<char32_t>(0xAC00)));  // 가
  }
  // LV + T -> LVT
  {
    std::u32string s = {static_cast<char32_t>(0xAC00), 0x11A8};
    NormalizeNfc(&s);
    EXPECT_EQ(s, std::u32string(1, static_cast<char32_t>(0xAC01)));  // 각
  }
  // Precomposed syllables round-trip.
  {
    std::u32string s = {static_cast<char32_t>(0xAC01)};
    NormalizeNfc(&s);
    EXPECT_EQ(s, std::u32string(1, static_cast<char32_t>(0xAC01)));
  }
}

TEST(NormalizeNfc, MultiStageComposition) {
  // â + COMBINING ACUTE -> U+1EA5. This is a two-part immediate decomposition
  // whose full expansion is three codepoints, so it is easy to drop when the
  // table is built from the wrong decomposition.
  std::u32string s = {0xE2, 0x301};
  NormalizeNfc(&s);
  EXPECT_EQ(s, std::u32string(1, static_cast<char32_t>(0x1EA5)));
}

TEST(NormalizeNfc, CompositionExclusionsAreNotComposed) {
  // NFC deliberately refuses these. They must stay decomposed.
  {
    // DEVANAGARI KA + NUKTA is not composed to U+0958.
    std::u32string s = {0x915, 0x93C};
    NormalizeNfc(&s);
    EXPECT_EQ(s.size(), 2u);
  }
}

TEST(NormalizeLexiconKey, MatchesAcrossDecompositionAndCase) {
  // The lexicon entry and the query may use different forms; both must land
  // on the same key.
  EXPECT_EQ(NormalizeLexiconKey(std::string("J\xcc\x8c")),  // J + caron
            NormalizeLexiconKey(std::string("\xc7\xb0")));  // ǰ
  EXPECT_EQ(NormalizeLexiconKey(std::string("J\xcc\x8c")), std::string("\xc7\xb0"));
}

TEST(SplitLexiconPieces, HyphenJoinsPieces) {
  // "two-thirds" is one spoken word: only the first piece takes a boundary.
  auto pieces = SplitLexiconPieces(Utf8ToUtf32("two-thirds"));
  ASSERT_EQ(pieces.size(), 2u);
  EXPECT_EQ(pieces[0].text, "two");
  EXPECT_TRUE(pieces[0].needs_boundary);
  EXPECT_EQ(pieces[1].text, "thirds");
  EXPECT_FALSE(pieces[1].needs_boundary);
}

TEST(SplitLexiconPieces, EmDashSplitsPieces) {
  // An em dash is a word separator: both pieces take a boundary, so the two
  // pronunciations are not run together.
  auto pieces = SplitLexiconPieces(Utf8ToUtf32("hello—world"));
  ASSERT_EQ(pieces.size(), 2u);
  EXPECT_EQ(pieces[0].text, "hello");
  EXPECT_TRUE(pieces[0].needs_boundary);
  EXPECT_EQ(pieces[1].text, "world");
  EXPECT_TRUE(pieces[1].needs_boundary);
}

TEST(SplitLexiconPieces, SinglePiece) {
  auto pieces = SplitLexiconPieces(Utf8ToUtf32("Dveme"));
  ASSERT_EQ(pieces.size(), 1u);
  EXPECT_EQ(pieces[0].text, "Dveme");
  EXPECT_TRUE(pieces[0].needs_boundary);
}

TEST(SplitLexiconPieces, DigitsSurviveSplitting) {
  // Pinyin keys carry tone digits; they are word characters and must not be
  // trimmed away when a hyphenated token is split.
  auto pieces = SplitLexiconPieces(Utf8ToUtf32("zhong1-guo2"));
  ASSERT_EQ(pieces.size(), 2u);
  EXPECT_EQ(pieces[0].text, "zhong1");
  EXPECT_EQ(pieces[1].text, "guo2");
  EXPECT_FALSE(pieces[1].needs_boundary);
}

TEST(SplitLexiconPieces, MixedSeparators) {
  // "a-b—c": hyphen joins a/b, em dash starts a new word at c.
  auto pieces = SplitLexiconPieces(Utf8ToUtf32("a-b—c"));
  ASSERT_EQ(pieces.size(), 3u);
  EXPECT_EQ(pieces[0].text, "a");
  EXPECT_TRUE(pieces[0].needs_boundary);
  EXPECT_EQ(pieces[1].text, "b");
  EXPECT_FALSE(pieces[1].needs_boundary);
  EXPECT_EQ(pieces[2].text, "c");
  EXPECT_TRUE(pieces[2].needs_boundary);
}

TEST(SplitLexiconPieces, LeadingTrailingSeparatorsProduceNoEmptyPieces) {
  auto pieces = SplitLexiconPieces(Utf8ToUtf32("---word---"));
  ASSERT_EQ(pieces.size(), 1u);
  EXPECT_EQ(pieces[0].text, "word");
  EXPECT_TRUE(pieces[0].needs_boundary);
}

// ---------------------------------------------------------------------
// Follow-up review findings.
// ---------------------------------------------------------------------

TEST(ToSimpleLower, SimpleMappingNotFullLowercase) {
  // str.lower() maps U+0130 to i + U+0307 (two codepoints). The table must
  // hold the *simple* mapping, U+0130 -> i, or Istanbul never matches
  // istanbul.
  EXPECT_EQ(ToSimpleLower(static_cast<char32_t>(0x130)), U'i');
  // Cased numerals and symbols are not category L but still case-fold.
  EXPECT_EQ(ToSimpleLower(static_cast<char32_t>(0x2167)), 0x2177);  // Ⅷ -> ⅸ
  EXPECT_EQ(ToSimpleLower(static_cast<char32_t>(0x24B6)), 0x24D0);  // Ⓐ -> ⓐ
}

TEST(ToSimpleLower, ExpansionsAreNotMapped) {
  // These have no 1:1 simple lowercase: U+00DF expands to "ss" and U+FB01 to
  // "fi". Mapping them to their first character would be wrong.
  EXPECT_EQ(ToSimpleLower(static_cast<char32_t>(0xDF)), 0xDF);
  EXPECT_EQ(ToSimpleLower(static_cast<char32_t>(0xFB01)), 0xFB01);
}

TEST(NormalizeNfc, ComposesBeforeTokenizationWouldSplit) {
  // U+3099 is a combining voiced sound mark that IsCJK() also accepts, so a
  // tokenizer that splits before normalizing emits it as its own token and
  // the pair can never match a stored が. Normalizing the whole query first
  // keeps it as one character.
  std::u32string s = {0x304B, 0x3099};  // HIRAGANA KA + voiced mark
  NormalizeNfc(&s);
  EXPECT_EQ(s, std::u32string(1, static_cast<char32_t>(0x304C)));  // が
}

TEST(CountLexiconPhraseWords, CjkCountsOnePerCharacterAfterNfc) {
  // Once normalized, a decomposed Hangul pair is one syllable, hence one
  // lookup token; the Jamo form would count as two.
  std::string jamo = "\xe1\x84\x92\xe1\x85\xa1";  // U+1112 U+1161
  std::string syllable = NormalizeNfc(jamo);      // U+1112 + U+1161 -> U+D55C
  EXPECT_EQ(CountLexiconPhraseWords(jamo), 2);
  EXPECT_EQ(CountLexiconPhraseWords(syllable), 1);
}

TEST(AssignLexiconPieceBoundaries, SkippedPiecePassesItsBoundaryOn) {
  // "hello zzzqxx-world": zzzqxx is OOV and dropped, but it still owes a
  // boundary before "world", or the two spoken words run together.
  std::vector<bool> needs = {true, true, false};  // hello | zzzqxx-world
  std::vector<bool> found = {true, false, true};
  auto b = AssignLexiconPieceBoundaries(needs, found);
  ASSERT_EQ(b.size(), 3u);
  EXPECT_TRUE(b[0]);   // hello
  EXPECT_FALSE(b[1]);  // skipped
  EXPECT_TRUE(b[2]);   // world inherits zzzqxx's claim
}

TEST(AssignLexiconPieceBoundaries, HyphenDoesNotSplitEmittedPieces) {
  // "two-thirds": both found, hyphen joins them into one spoken word.
  std::vector<bool> needs = {true, false};
  std::vector<bool> found = {true, true};
  auto b = AssignLexiconPieceBoundaries(needs, found);
  ASSERT_EQ(b.size(), 2u);
  EXPECT_TRUE(b[0]);
  EXPECT_FALSE(b[1]);
}

TEST(AssignLexiconPieceBoundaries, TrailingSkippedPieceClaimsNothing) {
  // Nothing after the skipped piece to receive the boundary.
  std::vector<bool> needs = {true, true};
  std::vector<bool> found = {true, false};
  auto b = AssignLexiconPieceBoundaries(needs, found);
  ASSERT_EQ(b.size(), 2u);
  EXPECT_TRUE(b[0]);
  EXPECT_FALSE(b[1]);
}

TEST(AssignLexiconPieceBoundaries, ConsecutiveSkippedPiecesCollapseToOneBoundary) {
  std::vector<bool> needs = {true, true, false};
  std::vector<bool> found = {false, false, true};
  auto b = AssignLexiconPieceBoundaries(needs, found);
  ASSERT_EQ(b.size(), 3u);
  EXPECT_FALSE(b[0]);
  EXPECT_FALSE(b[1]);
  EXPECT_TRUE(b[2]);
}

TEST(ToLowerCase, BytesOverloadStaysByteWise) {
  // The legacy lexicon loaders (CharacterLexicon, HomophoneReplacer, Lexicon,
  // MeloTts, the NeMo transducer model) call this overload and do not apply
  // Unicode lowercasing when looking keys up. Making it Unicode-aware stored
  // fullwidth A as its small form and then failed to match an unchanged query.
  std::string s = "\xef\xbc\xa1";  // fullwidth A, U+FF21
  ToLowerCase(&s);
  EXPECT_EQ(s, std::string("\xef\xbc\xa1"));

  std::string ascii = "ABC";
  ToLowerCase(&ascii);
  EXPECT_EQ(ascii, "abc");
}

TEST(NormalizeNfc, MarksComposeAndThenBlock) {
  // a + dot-below(220) composes to U+1EA1; the acute(230) stays a mark on the
  // composite. Confirmed against Python's unicodedata NFC.
  std::u32string s = {U'a', 0x0323, 0x0301};
  NormalizeNfc(&s);
  ASSERT_EQ(s.size(), 2u);
  EXPECT_EQ(s[0], 0x1EA1);
  EXPECT_EQ(s[1], 0x0301);
}

TEST(NormalizeNfc, LeadingMarkIsNotACompositionTarget) {
  // A leading mark must not be treated as a starter: it must not absorb a
  // following mark, and it must count towards the blocking class.
  std::u32string s = {0x0301, 0x0301, U'a'};
  NormalizeNfc(&s);
  std::u32string want = {0x0301, 0x0301, U'a'};
  EXPECT_EQ(s, want);
}

TEST(NormalizeNfc, LongMarkRunIsNotQuadratic) {
  // Regression for the O(n^2) rescan: a base followed by a long run of marks.
  // At 50k marks the quadratic version needs ~2.5e9 backward-search steps and
  // would stall; the incremental one is linear. This test is the guard.
  std::u32string s(50001, 0x0301);
  s[0] = U'b';
  NormalizeNfc(&s);
  EXPECT_EQ(s[0], U'b');
}

// Mark names are spelled out because raw codepoints in a table are impossible
// to review. Expected values come from Python's unicodedata.normalize('NFC'),
// the reference implementation.
static constexpr char32_t kAcute = 0x0301;        // combining acute accent
static constexpr char32_t kGrave = 0x0300;        // combining grave accent
static constexpr char32_t kMacron = 0x0304;       // combining macron
static constexpr char32_t kDotBelow = 0x0323;     // combining dot below
static constexpr char32_t kGraveBelow = 0x0316;   // combining grave below
static constexpr char32_t kCedilla = 0x0327;      // combining cedilla
static constexpr char32_t kVoicedMark = 0x3099;   // combining voiced sound mark

// Hangul jamo and syllables (UAX #15 section 16). The syllables are derived
// from the jamo rather than typed in, so they cannot go stale.
static constexpr char32_t kHangulL = 0x1112;   // choseong hieuh
static constexpr char32_t kHangulV = 0x1161;   // jungseong a
static constexpr char32_t kHangulT = 0x11A8;   // jongseong kiyeok
static constexpr char32_t kHangulLV =
    0xAC00 + ((kHangulL - 0x1100) * 21 + (kHangulV - 0x1161)) * 28;
static constexpr char32_t kHangulLVT = kHangulLV + (kHangulT - 0x11A7);

TEST(NormalizeNfc, MatchesUnicodeReference) {
  struct Case {
    const char *what;
    std::u32string in;
    std::u32string want;
  };
  static const Case cases[] = {
      // ---- base + mark, where a precomposed form exists ----
      {"a + acute composes", {U'a', kAcute}, {0x00E1}},
      {"A + acute composes", {U'A', kAcute}, {0x00C1}},
      {"a + macron composes", {U'a', kMacron}, {0x0101}},
      {"A + macron composes to U+0100", {U'A', kMacron}, {0x0100}},
      {"a + grave composes", {U'a', kGrave}, {0x00E0}},

      // ---- base + mark, where no precomposed form exists ----
      {"a + grave-below stays decomposed", {U'a', kGraveBelow}, {U'a', kGraveBelow}},
      {"a + cedilla stays decomposed", {U'a', kCedilla}, {U'a', kCedilla}},

      // ---- multi-stage: the pair is the immediate decomposition ----
      {"a-circumflex + acute -> U+1EA5", {0x00E2, kAcute}, {0x1EA5}},
      {"a + dot-below + acute", {U'a', kDotBelow, kAcute}, {0x1EA1, kAcute}},

      // ---- starter + starter (Bengali, Tamil) ----
      {"Bengali U+09C7 + U+09BE -> U+09CB", {0x09C7, 0x09BE}, {0x09CB}},
      {"Tamil U+0BC6 + U+0BBE -> U+0BCA", {0x0BC6, 0x0BBE}, {0x0BCA}},

      // ---- Hangul is algorithmic ----
      {"Hangul L + V -> LV", {kHangulL, kHangulV}, {kHangulLV}},
      {"Hangul LV + T -> LVT", {kHangulLV, kHangulT}, {kHangulLVT}},
      {"precomposed syllable round-trips", {kHangulLVT}, {kHangulLVT}},

      // ---- Japanese voiced mark, which IsCJK also accepts ----
      {"HIRAGANA KA + voiced mark -> GA", {0x304B, kVoicedMark}, {0x304C}},

      // ---- leading marks must not become composition targets ----
      {"leading mark is inert", {kAcute, U'a'}, {kAcute, U'a'}},
      {"two leading marks stay", {kAcute, kAcute, U'a'}, {kAcute, kAcute, U'a'}},

      // ---- equal combining class blocks (UAX #15 D117) ----
      {"a + acute + acute keeps the second", {U'a', kAcute, kAcute}, {0x00E1, kAcute}},

      // ---- already-composed input round-trips ----
      {"precomposed U+0101 round-trips", {0x0101}, {0x0101}},
      {"precomposed U+1EA5 round-trips", {0x1EA5}, {0x1EA5}},

      // ---- mixed ----
      {"macron then grave and acute", {U'a', kMacron, kGrave, kAcute}, {0x0101, kGrave, kAcute}},
  };

  for (const auto &c : cases) {
    std::u32string s = c.in;
    NormalizeNfc(&s);
    EXPECT_EQ(s, c.want) << c.what;
  }
}

}  // namespace sherpa_onnx
