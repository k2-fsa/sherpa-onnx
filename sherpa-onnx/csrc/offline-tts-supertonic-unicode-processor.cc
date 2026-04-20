// sherpa-onnx/csrc/offline-tts-supertonic-unicode-processor.cc
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#include "sherpa-onnx/csrc/offline-tts-supertonic-unicode-processor.h"

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {
namespace {

// Hangul syllable decomposition constants (Unicode Standard Annex #15)
static constexpr uint32_t kHangulSbase = 0xAC00;  // Start of Hangul syllables
static constexpr uint32_t kHangulLbase = 0x1100;  // Start of Hangul Jamo
static constexpr uint32_t kHangulVbase = 0x1161;  // Start of Hangul vowels
static constexpr uint32_t kHangulTbase = 0x11A7;  // Start of Hangul trailing
static constexpr int32_t kHangulLcount = 19;
static constexpr int32_t kHangulVcount = 21;
static constexpr int32_t kHangulTcount = 28;
static constexpr int32_t kHangulNcount = kHangulVcount * kHangulTcount;  // 588
static constexpr int32_t kHangulScount =
    kHangulLcount * kHangulNcount;  // 11172  // NOLINT

// Latin NFKD decompositions via switch (no static map allocation).
// Returns true if codepoint was decomposed, false otherwise.
static bool DecomposeLatin(uint32_t codepoint, std::vector<uint16_t> *out) {
  auto push2 = [&](uint16_t a, uint16_t b) {
    out->push_back(a);
    out->push_back(b);
  };
  switch (codepoint) {
    case 0x00C1:
      push2(0x0041, 0x0301);
      return true;
    case 0x00C9:
      push2(0x0045, 0x0301);
      return true;
    case 0x00CD:
      push2(0x0049, 0x0301);
      return true;
    case 0x00D3:
      push2(0x004F, 0x0301);
      return true;
    case 0x00DA:
      push2(0x0055, 0x0301);
      return true;
    case 0x00E1:
      push2(0x0061, 0x0301);
      return true;
    case 0x00E9:
      push2(0x0065, 0x0301);
      return true;
    case 0x00ED:
      push2(0x0069, 0x0301);
      return true;
    case 0x00F3:
      push2(0x006F, 0x0301);
      return true;
    case 0x00FA:
      push2(0x0075, 0x0301);
      return true;
    case 0x00C0:
      push2(0x0041, 0x0300);
      return true;
    case 0x00C8:
      push2(0x0045, 0x0300);
      return true;
    case 0x00CC:
      push2(0x0049, 0x0300);
      return true;
    case 0x00D2:
      push2(0x004F, 0x0300);
      return true;
    case 0x00D9:
      push2(0x0055, 0x0300);
      return true;
    case 0x00E0:
      push2(0x0061, 0x0300);
      return true;
    case 0x00E8:
      push2(0x0065, 0x0300);
      return true;
    case 0x00EC:
      push2(0x0069, 0x0300);
      return true;
    case 0x00F2:
      push2(0x006F, 0x0300);
      return true;
    case 0x00F9:
      push2(0x0075, 0x0300);
      return true;
    case 0x00C2:
      push2(0x0041, 0x0302);
      return true;
    case 0x00CA:
      push2(0x0045, 0x0302);
      return true;
    case 0x00CE:
      push2(0x0049, 0x0302);
      return true;
    case 0x00D4:
      push2(0x004F, 0x0302);
      return true;
    case 0x00DB:
      push2(0x0055, 0x0302);
      return true;
    case 0x00E2:
      push2(0x0061, 0x0302);
      return true;
    case 0x00EA:
      push2(0x0065, 0x0302);
      return true;
    case 0x00EE:
      push2(0x0069, 0x0302);
      return true;
    case 0x00F4:
      push2(0x006F, 0x0302);
      return true;
    case 0x00FB:
      push2(0x0075, 0x0302);
      return true;
    case 0x00C3:
      push2(0x0041, 0x0303);
      return true;
    case 0x00D1:
      push2(0x004E, 0x0303);
      return true;
    case 0x00D5:
      push2(0x004F, 0x0303);
      return true;
    case 0x00E3:
      push2(0x0061, 0x0303);
      return true;
    case 0x00F1:
      push2(0x006E, 0x0303);
      return true;
    case 0x00F5:
      push2(0x006F, 0x0303);
      return true;
    case 0x00C4:
      push2(0x0041, 0x0308);
      return true;
    case 0x00CB:
      push2(0x0045, 0x0308);
      return true;
    case 0x00CF:
      push2(0x0049, 0x0308);
      return true;
    case 0x00D6:
      push2(0x004F, 0x0308);
      return true;
    case 0x00DC:
      push2(0x0055, 0x0308);
      return true;
    case 0x00E4:
      push2(0x0061, 0x0308);
      return true;
    case 0x00EB:
      push2(0x0065, 0x0308);
      return true;
    case 0x00EF:
      push2(0x0069, 0x0308);
      return true;
    case 0x00F6:
      push2(0x006F, 0x0308);
      return true;
    case 0x00FC:
      push2(0x0075, 0x0308);
      return true;
    case 0x00C7:
      push2(0x0043, 0x0327);
      return true;
    case 0x00E7:
      push2(0x0063, 0x0327);
      return true;
    default:
      return false;
  }
}

static void DecomposeCharacter(uint32_t codepoint,
                               std::vector<uint16_t> *output) {
  if (codepoint >= kHangulSbase && codepoint < kHangulSbase + kHangulScount) {
    uint32_t s_index = codepoint - kHangulSbase;
    uint32_t l_index = s_index / kHangulNcount;
    uint32_t v_index = (s_index % kHangulNcount) / kHangulTcount;
    uint32_t t_index = s_index % kHangulTcount;

    output->push_back(static_cast<uint16_t>(kHangulLbase + l_index));
    output->push_back(static_cast<uint16_t>(kHangulVbase + v_index));
    if (t_index > 0) {
      output->push_back(static_cast<uint16_t>(kHangulTbase + t_index));
    }
    return;
  }

  if (DecomposeLatin(codepoint, output)) return;

  if (codepoint > 0xFFFF) return;
  output->push_back(static_cast<uint16_t>(codepoint));
}

// Decode the last UTF-8 codepoint in s. Returns 0 if s is empty or invalid.
static uint32_t LastCodepointUtf8(const std::string &s) {
  if (s.empty()) return 0;

  size_t start = s.size() - 1;
  while (start > 0 && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) {
    --start;
  }

  unsigned char c = static_cast<unsigned char>(s[start]);

  if ((c & 0x80) == 0) return c;

  if ((c & 0xE0) == 0xC0 && start + 1 < s.size()) {
    return ((c & 0x1F) << 6) |
           (static_cast<unsigned char>(s[start + 1]) & 0x3F);
  }

  if ((c & 0xF0) == 0xE0 && start + 2 < s.size()) {
    return ((c & 0x0F) << 12) |
           ((static_cast<unsigned char>(s[start + 1]) & 0x3F) << 6) |
           (static_cast<unsigned char>(s[start + 2]) & 0x3F);
  }

  if ((c & 0xF8) == 0xF0 && start + 3 < s.size()) {
    return ((c & 0x07) << 18) |
           ((static_cast<unsigned char>(s[start + 1]) & 0x3F) << 12) |
           ((static_cast<unsigned char>(s[start + 2]) & 0x3F) << 6) |
           (static_cast<unsigned char>(s[start + 3]) & 0x3F);
  }

  return 0;
}

static bool IsEndingPunctuationCodepoint(uint32_t cp) {
  switch (cp) {
    case 0x2026:  // …
    case 0x3002:  // 。
    case 0x300D:  // 」
    case 0x300F:  // 』
    case 0x3011:  // 】
    case 0x3009:  // 〉
    case 0x300B:  // 》
    case 0x203A:  // ›
    case 0x00BB:  // »
    case 0x201C:  // "
    case 0x201D:  // "
    case 0x2018:  // '
    case 0x2019:  // '
      return true;
    default:
      return false;
  }
}

static void ReplaceString(std::string *text, const std::string &from,
                          const std::string &to) {
  size_t pos = 0;
  while ((pos = text->find(from, pos)) != std::string::npos) {
    text->replace(pos, from.length(), to);
    pos += to.length();
  }
}

// Load indexer from raw int32_t binary (from generate_indexer_bin.py).
static std::vector<int32_t> LoadIndexerFromBinary(const char *data,
                                                  size_t size) {
  if (size == 0 || (size % sizeof(int32_t) != 0)) {
    SHERPA_ONNX_LOGE(
        "Invalid unicode indexer .bin size: %zu (must be multiple of %zu)",
        size, sizeof(int32_t));
    SHERPA_ONNX_EXIT(-1);
  }
  size_t count = size / sizeof(int32_t);
  std::vector<int32_t> out(count);
  std::memcpy(out.data(), data, size);
  return out;
}

static std::vector<int32_t> LoadIndexerFromPathImpl(
    const std::vector<char> &buf, const std::string &path) {
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Failed to read unicode indexer: %s", path.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  return LoadIndexerFromBinary(buf.data(), buf.size());
}

}  // namespace

SupertonicUnicodeProcessor::SupertonicUnicodeProcessor(
    const std::string &unicode_indexer_path) {
  if (!EndsWith(unicode_indexer_path, ".bin")) {
    SHERPA_ONNX_LOGE("Unicode indexer path must be end with .bin. Given: '%s'",
                     unicode_indexer_path.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  std::vector<char> buf = ReadFile(unicode_indexer_path);
  indexer_ = LoadIndexerFromPathImpl(buf, unicode_indexer_path);
}

template <typename Manager>
SupertonicUnicodeProcessor::SupertonicUnicodeProcessor(
    Manager *mgr, const std::string &unicode_indexer_path) {
  if (!EndsWith(unicode_indexer_path, ".bin")) {
    SHERPA_ONNX_LOGE("Unicode indexer path must be end with .bin. Given: '%s'",
                     unicode_indexer_path.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  std::vector<char> buf = ReadFile(mgr, unicode_indexer_path);
  indexer_ = LoadIndexerFromPathImpl(buf, unicode_indexer_path);
}

std::string SupertonicUnicodeProcessor::PreprocessText(
    const std::string &text, const std::string &lang) const {
  std::string result = text;

  static constexpr std::array<std::pair<const char *, const char *>, 25>
      replacements = {{
          {"–", "-"},
          {"‑", "-"},
          {"—", "-"},
          {"_", " "},
          {u8"\u201C", "\""},
          {u8"\u201D", "\""},
          {u8"\u2018", "'"},
          {u8"\u2019", "'"},
          {"´", "'"},
          {"`", "'"},
          {"[", " "},
          {"]", " "},
          {"|", " "},
          {"/", " "},
          {"#", " "},
          {"→", " "},
          {"←", " "},
          {"♥", ""},
          {"☆", ""},
          {"♡", ""},
          {"©", ""},
          {"\\", ""},
          {"@", " at "},
          {"e.g.,", "for example, "},
          {"i.e.,", "that is, "},
      }};

  for (const auto &repl : replacements) {
    ReplaceString(&result, repl.first, repl.second);
  }

  // Remove some U+1Fxxx emoji/symbols (4-byte UTF-8 sequences: F0 9F 80-BF
  // 80-BF). Note: This only removes a subset of emoji (U+1F000-U+1FFFF), not
  // all emoji. Optimized: manual scanning instead of regex.
  std::string emoji_removed;
  emoji_removed.reserve(result.size());
  for (size_t i = 0; i < result.size();) {
    if (i + 3 < result.size() &&
        static_cast<unsigned char>(result[i]) == 0xF0 &&
        static_cast<unsigned char>(result[i + 1]) == 0x9F &&
        (static_cast<unsigned char>(result[i + 2]) & 0xC0) == 0x80 &&
        (static_cast<unsigned char>(result[i + 3]) & 0xC0) == 0x80) {
      i += 4;  // Skip emoji
    } else {
      emoji_removed += result[i];
      ++i;
    }
  }
  result = std::move(emoji_removed);

  // Fix spacing around punctuation (optimized: single pass)
  std::string punct_fixed;
  punct_fixed.reserve(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i] == ' ' && i + 1 < result.size()) {
      char next = result[i + 1];
      if (next == ',' || next == '.' || next == '!' || next == '?' ||
          next == ';' || next == ':' || next == '\'') {
        punct_fixed += next;
        ++i;  // Skip space and punctuation
        continue;
      }
    }
    punct_fixed += result[i];
  }
  result = std::move(punct_fixed);

  // Collapse adjacent duplicate quotes ("" -> ", '' -> ') while preserving
  // normal paired quotes. Discard backticks. Single-pass O(n) algorithm.
  std::string quotes_fixed;
  quotes_fixed.reserve(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i] == '`') {
      // Skip backticks
      continue;
    }
    if (result[i] == '"' && i + 1 < result.size() && result[i + 1] == '"') {
      // Collapse adjacent double quotes: "" -> "
      quotes_fixed += '"';
      ++i;  // Skip the second quote
    } else if (result[i] == '\'' && i + 1 < result.size() &&
               result[i + 1] == '\'') {
      // Collapse adjacent single quotes: '' -> '
      quotes_fixed += '\'';
      ++i;  // Skip the second quote
    } else {
      quotes_fixed += result[i];
    }
  }
  result = std::move(quotes_fixed);

  // Remove extra spaces (optimized: single pass)
  std::string spaces_fixed;
  spaces_fixed.reserve(result.size());
  bool last_was_space = false;
  for (char c : result) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (!last_was_space) {
        spaces_fixed += ' ';
        last_was_space = true;
      }
    } else {
      spaces_fixed += c;
      last_was_space = false;
    }
  }
  result = Trim(spaces_fixed);

  if (!result.empty()) {
    char last_char = result.back();
    bool ends_with_punct =
        (last_char == '.' || last_char == '!' || last_char == '?' ||
         last_char == ';' || last_char == ':' || last_char == ',' ||
         last_char == '\'' || last_char == '"' || last_char == ')' ||
         last_char == ']' || last_char == '}' || last_char == '>');
    if (!ends_with_punct) {
      ends_with_punct = IsEndingPunctuationCodepoint(LastCodepointUtf8(result));
    }
    if (!ends_with_punct) {
      result += ".";
    }
  }

  // Wrap text with language tags
  result = "<" + lang + ">" + result + "</" + lang + ">";

  return result;
}

std::vector<uint16_t> SupertonicUnicodeProcessor::TextToUnicodeValues(
    const std::string &text) const {
  std::vector<uint16_t> unicode_values;
  size_t i = 0;

  while (i < text.size()) {
    uint32_t codepoint = 0;
    unsigned char c = static_cast<unsigned char>(text[i]);

    if ((c & 0x80) == 0) {
      codepoint = c;
      i += 1;
    } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
      codepoint = (c & 0x1F) << 6;
      codepoint |= (static_cast<unsigned char>(text[i + 1]) & 0x3F);
      i += 2;
    } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
      codepoint = (c & 0x0F) << 12;
      codepoint |= (static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6;
      codepoint |= (static_cast<unsigned char>(text[i + 2]) & 0x3F);
      i += 3;
    } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
      codepoint = (c & 0x07) << 18;
      codepoint |= (static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12;
      codepoint |= (static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6;
      codepoint |= (static_cast<unsigned char>(text[i + 3]) & 0x3F);
      i += 4;
    } else {
      i += 1;
      continue;
    }

    DecomposeCharacter(codepoint, &unicode_values);
  }

  return unicode_values;
}

void SupertonicUnicodeProcessor::Process(
    const std::string &text, const std::string &lang,
    std::vector<int64_t> *text_ids, std::vector<float> *text_mask_flat,
    std::vector<int64_t> *text_mask_shape) const {
  const std::string processed = PreprocessText(text, lang);
  const std::vector<uint16_t> unicode_vals = TextToUnicodeValues(processed);
  const size_t seq_len = unicode_vals.size();

  constexpr int64_t kUnknownId = 0;
  text_ids->assign(seq_len, kUnknownId);
  for (size_t i = 0; i < seq_len; ++i) {
    const size_t u = unicode_vals[i];
    (*text_ids)[i] = (u < indexer_.size()) ? indexer_[u] : kUnknownId;
  }

  // Batch size is always 1: mask is all ones, shape [1, 1, seq_len].
  text_mask_flat->assign(seq_len, 1.0f);
  text_mask_shape->assign({1, 1, static_cast<int64_t>(seq_len)});
}

#if __ANDROID_API__ >= 9
template SupertonicUnicodeProcessor::SupertonicUnicodeProcessor(
    AAssetManager *mgr, const std::string &unicode_indexer_path);
#endif

#if __OHOS__
template SupertonicUnicodeProcessor::SupertonicUnicodeProcessor(
    NativeResourceManager *mgr, const std::string &unicode_indexer_path);
#endif

}  // namespace sherpa_onnx
