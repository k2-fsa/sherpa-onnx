// sherpa-onnx/csrc/text-utils.h
//
// Copyright 2009-2011  Saarland University;  Microsoft Corporation
// Copyright      2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_TEXT_UTILS_H_
#define SHERPA_ONNX_CSRC_TEXT_UTILS_H_
#include <errno.h>
#include <stdlib.h>

#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _MSC_VER
#define SHERPA_ONNX_STRTOLL(cur_cstr, end_cstr) \
  _strtoi64(cur_cstr, end_cstr, 10);
#else
#define SHERPA_ONNX_STRTOLL(cur_cstr, end_cstr) strtoll(cur_cstr, end_cstr, 10);
#endif

// This file is copied/modified from
// https://github.com/kaldi-asr/kaldi/blob/master/src/util/text-utils.h

namespace sherpa_onnx {

// Return a JSON string literal, including the surrounding quotes. Escape
// control characters, quotes, and backslashes; preserve UTF-8 bytes unchanged.
std::string QuoteJsonString(const std::string &text);

/// Converts a string into an integer via strtoll and returns false if there was
/// any kind of problem (i.e. the string was not an integer or contained extra
/// non-whitespace junk, or the integer was too large to fit into the type it is
/// being converted into).  Only sets *out if everything was OK and it returns
/// true.
template <class Int>
bool ConvertStringToInteger(const std::string &str, Int *out) {
  // copied from kaldi/src/util/text-util.h
  static_assert(std::is_integral<Int>::value, "");
  const char *this_str = str.c_str();
  char *end = nullptr;
  errno = 0;
  int64_t i = SHERPA_ONNX_STRTOLL(this_str, &end);
  if (end != this_str) {
    while (isspace(*end)) ++end;
  }
  if (end == this_str || *end != '\0' || errno != 0) return false;
  Int iInt = static_cast<Int>(i);
  if (static_cast<int64_t>(iInt) != i ||
      (i < 0 && !std::numeric_limits<Int>::is_signed)) {
    return false;
  }
  *out = iInt;
  return true;
}

/// Split a string using any of the single character delimiters.
/// If omit_empty_strings == true, the output will contain any
/// nonempty strings after splitting on any of the
/// characters in the delimiter.  If omit_empty_strings == false,
/// the output will contain n+1 strings if there are n characters
/// in the set "delim" within the input string.  In this case
/// the empty string is split to a single empty string.
void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out);

/// Trim leading and trailing whitespace from a string.
std::string Trim(const std::string &str);

/// Split a string by a single character delimiter, trim whitespace from each
/// part, and remove empty strings. This is a convenience wrapper around
/// SplitStringToVector with trimming and filtering.
std::vector<std::string> SplitStringAndTrim(const std::string &str, char delim);

/**
  \brief Split a string (e.g. 1:2:3) into a vector of integers.

  \param [in]  delim  String containing a list of characters, any of which
                      is allowed as a delimiter.
  \param [in] omit_empty_strings If true, empty strings between delimiters are
                      allowed and will not produce an output integer; if false,
                      instances of characters in 'delim' that are consecutive or
                      at the start or end of the string would be an error.
                      You'll normally want this to be true if 'delim' consists
                      of spaces, and false otherwise.
  \param [out] out   The output list of integers.
*/
template <class I>
bool SplitStringToIntegers(const std::string &full, const char *delim,
                           bool omit_empty_strings,  // typically false [but
                                                     // should probably be true
                                                     // if "delim" is spaces].
                           std::vector<I> *out) {
  static_assert(std::is_integral<I>::value, "");
  if (*(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, omit_empty_strings, &split);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); i++) {
    const char *this_str = split[i].c_str();
    char *end = NULL;
    int64_t j = 0;
    j = SHERPA_ONNX_STRTOLL(this_str, &end);
    if (end == this_str || *end != '\0') {
      out->clear();
      return false;
    } else {
      I jI = static_cast<I>(j);
      if (static_cast<int64_t>(jI) != j) {
        // output type cannot fit this integer.
        out->clear();
        return false;
      }
      (*out)[i] = jI;
    }
  }
  return true;
}

// This is defined for F = float and double.
template <class F>
bool SplitStringToFloats(const std::string &full, const char *delim,
                         bool omit_empty_strings,  // typically false
                         std::vector<F> *out);

// This is defined for F = float and double.
template <typename T>
bool ConvertStringToReal(const std::string &str, T *out);

std::vector<std::string> SplitUtf8(const std::string &text);

std::string ToLowerCase(const std::string &s);

// The single transform applied to both lexicon keys and lookup queries:
// Unicode NFC first, then lowercase. Every consumer of ParseLexiconFile --
// VITS, Kokoro, Kitten, Matcha, ZipVoice -- must use this for queries too,
// otherwise a normalized stored key will not match a decomposed query.
// Unicode handling is delegated to utf8proc so we do not maintain case or
// composition tables ourselves.

// Simple (1:1) Unicode lowercase mapping. Returns cp unchanged when there is
// no single-codepoint lowercase form (e.g. U+00DF LATIN SMALL LETTER SHARP S).
char32_t ToSimpleLower(char32_t cp);

// True if cp is a Unicode Letter (category L*) or Mark (category M*) or Number
// (category N*), i.e. something that can appear inside a lexicon word. Used to
// strip/split punctuation away from OOV tokens before retrying the lexicon.
// Numbers count so pinyin keys like "zhong1" keep their tone digits.
bool IsLexiconWordChar(char32_t cp);

// Unicode NFC. Needed so that e.g. "a" + U+0301 matches a lexicon entry stored
// as U+00E1. Returns true if |s| was modified.
bool NormalizeNfc(std::u32string *s);
std::string NormalizeNfc(const std::string &s);

// The single transform applied to both lexicon keys and lookup queries
std::string NormalizeLexiconKey(const std::string &s);

// Number of lookup tokens in a lexicon key. CJK counts one per character,
// and a space, a punctuation mark or a plain word run each counts one -- which
// is how Kokoro, Kitten and Matcha tokenize. Deliberately an over-estimate for
// VITS (it drops spaces) rather than an under-estimate: every longest-match
// loop clamps with `n - i`, so a too-large bound is harmless while a too-small
// one silently skips multi-token entries.
int32_t CountLexiconPhraseWords(const std::string &key);

// One lookup piece produced by splitting an OOV token on non-word characters.
struct LexiconPiece {
  std::string text;
  // True when a word boundary belongs in front of this piece. A hyphen joins
  // its pieces into one spoken word ("two-thirds" -> "two" + "thirds"); any
  // other separator starts a new word ("hello" + EM DASH + "world" is two).
  bool needs_boundary;
};

// Split |core| into lookup pieces for the OOV fallback. |core| is assumed to
// already have its leading/trailing non-word characters stripped.
std::vector<LexiconPiece> SplitLexiconPieces(const std::u32string &core);

// Decide which lookup pieces get a word boundary in front. |needs_boundary|
// and |found| are parallel to the piece list; |found[i]| is false when the
// piece has no lexicon entry and is skipped. A skipped piece passes its
// boundary claim on to the next emitted one, so dropping a word cannot glue
// the words around it together.
std::vector<bool> AssignLexiconPieceBoundaries(
    const std::vector<bool> &needs_boundary, const std::vector<bool> &found);
void ToLowerCase(std::string *in_out);


std::string RemoveInvalidUtf8Sequences(const std::string &text,
                                       bool show_debug_msg = false);

// Return true if text contains valid utf8 sequence.
// Return false otherwise
bool IsUtf8(const std::string &text);

// Return true if text contains valid gb2312 encoded sequence
// Return false otherwise
bool IsGB2312(const std::string &text);

#if defined(_WIN32)
std::string Gb2312ToUtf8(const std::string &text);
#endif

std::wstring ToWideString(const std::string &s);

std::string ToString(const std::wstring &s);

bool EndsWith(const std::string &haystack, const std::string &needle);

bool Contains(const std::string &haystack, const std::string &needle);

std::vector<std::string> SplitString(const std::string &s, int32_t chunk_size);

std::string Join(const std::vector<std::string> &ss,
                 const std::string &delim = "");

// Converts a UTF-8 std::string to a UTF-32 std::u32string
std::u32string Utf8ToUtf32(const std::string &str);

// Converts a UTF-8 std::string to a UTF-16 std::u16string. A code point
// above U+FFFF becomes a surrogate pair; invalid bytes become U+FFFD.
std::u16string Utf8ToUtf16(const std::string &str);

// Converts a UTF-32 std::u32string to a UTF-8 std::string
std::string Utf32ToUtf8(const std::u32string &str);

// Converts a single UTF-32 codepoint to a UTF-8 std::string
std::string Utf32ToUtf8(char32_t cp);

// Converts a UTF-16 std::u16string to a UTF-8 std::string.
// Surrogate pairs are combined; an unpaired surrogate becomes U+FFFD.
std::string Utf16ToUtf8(const std::u16string &str);

// Helper: Convert ASCII chars in a std::string to uppercase (leaves non-ASCII
// unchanged)
std::string ToUpperAscii(const std::string &str);

// Helper: Convert ASCII chars in a std::string to lowercase (leaves non-ASCII
// unchanged)
std::string ToLowerAscii(const std::string &str);

bool IsAlphaOrPunct(int ch);

// Detect if a codepoint is a CJK character
bool IsCJK(char32_t cp);

bool ContainsCJK(const std::string &text);

bool ContainsCJK(const std::u32string &text);

std::string RemoveSpaceBetweenCjk(const std::string &text);

std::string RemoveLeadingSpaces(const std::string &text);

// If the first character is a punctuation, drop it.
std::string RemoveLeadingPunct(const std::string &text);

bool StringToBool(const std::string &s);

// end is inclusive
std::string GetWord(const std::vector<std::string> &words, int32_t start,
                    int32_t end);

bool IsPunct(const std::string &s);

#if defined(_WIN32)
#define SHERPA_ONNX_TO_ORT_PATH(s) (ToWideString(s).c_str())
#else
#define SHERPA_ONNX_TO_ORT_PATH(s) ((s).c_str())
#endif

int32_t ToIntOrDefault(const std::string &s, int32_t default_value);

float ToFloatOrDefault(const std::string &s, float default_value);

// Convert lengths to flat mask + shape. Outputs [batch, 1, max_len] format
// where mask[b][0][i] = 1.0 if i < lengths[b], else 0.0.
void LengthsToMask(const std::vector<int64_t> &lengths,
                   std::vector<float> *mask_flat,
                   std::vector<int64_t> *mask_shape);

// TTS text chunking helpers.
std::vector<std::string> SplitByBlankLines(const std::string &text);
std::vector<std::string> SplitByPunctuation(const std::string &text);
// Split text at all punctuation boundaries: sentence-ending (.!?) and
// mid-sentence (,;: and their Chinese equivalents).
std::vector<std::string> SplitByAllPunctuation(const std::string &text);
std::vector<std::string> MergeShortSentences(
    const std::vector<std::string> &sentences, size_t min_chars);
std::vector<std::string> SplitLongSentence(const std::string &sentence,
                                           size_t max_chars);
std::vector<std::string> ChunkText(const std::string &text, size_t max_len);

// Count words in text. Each CJK character counts as one word.
// English words are sequences of non-CJK, non-space, non-punctuation chars.
int32_t CountWords(const std::string &text);

// Word-based versions of MergeShortSentences and SplitLongSentence.
// min_words/max_words are word counts (not codepoint counts).
// Each CJK character is one word; English words are space-separated.
std::vector<std::string> MergeShortSentencesByWords(
    const std::vector<std::string> &sentences, size_t min_words);
std::vector<std::string> SplitLongSentenceByWords(const std::string &sentence,
                                                  size_t max_words);

// A single lexicon entry: word (lowercased) and its phonemes as strings.
struct LexiconEntry {
  std::string key;                    // lowercased word/phrase
  std::vector<std::string> phonemes;  // phoneme strings (IPA or token names)
};

// Parse a lexicon file. Supports both formats:
//   - New-format (||): word1 word2 || phone1 phone2 phone3 ...
//   - Old-format (space): word phone1 phone2 ...
// Returns all entries and the max phrase length (in words/tokens).
std::vector<LexiconEntry> ParseLexiconFile(std::istream &is,
                                           int32_t *max_phrase_len_out);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TEXT_UTILS_H_
