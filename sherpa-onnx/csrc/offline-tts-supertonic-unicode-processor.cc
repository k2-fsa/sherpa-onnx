// sherpa-onnx/csrc/offline-tts-supertonic-unicode-processor.cc
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#include "sherpa-onnx/csrc/offline-tts-supertonic-unicode-processor.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <unordered_map>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

using json = nlohmann::json;

static const std::vector<std::string> kSupertonicAvailableLangs = {
    "en", "ko", "es", "pt", "fr"};

namespace {

// Hangul syllable decomposition constants (Unicode Standard Annex #15)
static constexpr uint32_t kHangulSbase = 0xAC00;  // Start of Hangul syllables
static constexpr uint32_t kHangulLbase = 0x1100;  // Start of Hangul Jamo
static constexpr uint32_t kHangulVbase = 0x1161;  // Start of Hangul vowels
static constexpr uint32_t kHangulTbase = 0x11A7;  // Start of Hangul trailing
static constexpr int kHangulLcount = 19;
static constexpr int kHangulVcount = 21;
static constexpr int kHangulTcount = 28;
static constexpr int kHangulNcount = kHangulVcount * kHangulTcount;  // 588
static constexpr int kHangulScount = kHangulLcount * kHangulNcount;  // 11172

// Latin character NFKD decompositions for Spanish, Portuguese, French
static const std::unordered_map<uint32_t, std::vector<uint16_t>>
    kLatinDecompositions = {
        // Acute accent
        {0x00C1, {0x0041, 0x0301}},
        {0x00C9, {0x0045, 0x0301}},
        {0x00CD, {0x0049, 0x0301}},
        {0x00D3, {0x004F, 0x0301}},
        {0x00DA, {0x0055, 0x0301}},
        {0x00E1, {0x0061, 0x0301}},
        {0x00E9, {0x0065, 0x0301}},
        {0x00ED, {0x0069, 0x0301}},
        {0x00F3, {0x006F, 0x0301}},
        {0x00FA, {0x0075, 0x0301}},
        // Grave accent
        {0x00C0, {0x0041, 0x0300}},
        {0x00C8, {0x0045, 0x0300}},
        {0x00CC, {0x0049, 0x0300}},
        {0x00D2, {0x004F, 0x0300}},
        {0x00D9, {0x0055, 0x0300}},
        {0x00E0, {0x0061, 0x0300}},
        {0x00E8, {0x0065, 0x0300}},
        {0x00EC, {0x0069, 0x0300}},
        {0x00F2, {0x006F, 0x0300}},
        {0x00F9, {0x0075, 0x0300}},
        // Circumflex
        {0x00C2, {0x0041, 0x0302}},
        {0x00CA, {0x0045, 0x0302}},
        {0x00CE, {0x0049, 0x0302}},
        {0x00D4, {0x004F, 0x0302}},
        {0x00DB, {0x0055, 0x0302}},
        {0x00E2, {0x0061, 0x0302}},
        {0x00EA, {0x0065, 0x0302}},
        {0x00EE, {0x0069, 0x0302}},
        {0x00F4, {0x006F, 0x0302}},
        {0x00FB, {0x0075, 0x0302}},
        // Tilde
        {0x00C3, {0x0041, 0x0303}},
        {0x00D1, {0x004E, 0x0303}},
        {0x00D5, {0x004F, 0x0303}},
        {0x00E3, {0x0061, 0x0303}},
        {0x00F1, {0x006E, 0x0303}},
        {0x00F5, {0x006F, 0x0303}},
        // Diaeresis
        {0x00C4, {0x0041, 0x0308}},
        {0x00CB, {0x0045, 0x0308}},
        {0x00CF, {0x0049, 0x0308}},
        {0x00D6, {0x004F, 0x0308}},
        {0x00DC, {0x0055, 0x0308}},
        {0x00E4, {0x0061, 0x0308}},
        {0x00EB, {0x0065, 0x0308}},
        {0x00EF, {0x0069, 0x0308}},
        {0x00F6, {0x006F, 0x0308}},
        {0x00FC, {0x0075, 0x0308}},
        // Cedilla
        {0x00C7, {0x0043, 0x0327}},
        {0x00E7, {0x0063, 0x0327}},
};

static void DecomposeCharacter(uint32_t codepoint,
                               std::vector<uint16_t> *output) {
  // Check Hangul syllables first
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

  // Check Latin decompositions
  auto it = kLatinDecompositions.find(codepoint);
  if (it != kLatinDecompositions.end()) {
    for (uint16_t cp : it->second) {
      output->push_back(cp);
    }
    return;
  }

  // Supertonic indexer only supports BMP (Basic Multilingual Plane,
  // U+0000-U+FFFF). Non-BMP codepoints (U+10000 and above) are discarded.
  if (codepoint > 0xFFFF) {
    return;
  }

  // BMP codepoint: keep as-is
  output->push_back(static_cast<uint16_t>(codepoint));
}

static void ReplaceString(std::string *text, const std::string &from,
                          const std::string &to) {
  size_t pos = 0;
  while ((pos = text->find(from, pos)) != std::string::npos) {
    text->replace(pos, from.length(), to);
    pos += to.length();
  }
}

static std::vector<int64_t> ParseIndexerFromJson(const json &j) {
  return j.get<std::vector<int64_t>>();
}
}  // namespace

SupertonicUnicodeProcessor::SupertonicUnicodeProcessor(
    const std::string &unicode_indexer_path) {
  json j = LoadJsonFromFile(unicode_indexer_path);
  indexer_ = ParseIndexerFromJson(j);
}

template <typename Manager>
SupertonicUnicodeProcessor::SupertonicUnicodeProcessor(
    Manager *mgr, const std::string &unicode_indexer_path) {
  auto buf = ReadFile(mgr, unicode_indexer_path);
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Failed to read unicode_indexer.json: %s",
                     unicode_indexer_path.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  json j = LoadJsonFromBuffer(buf);
  indexer_ = ParseIndexerFromJson(j);
}

std::string SupertonicUnicodeProcessor::PreprocessText(
    const std::string &text, const std::string &lang) const {
  std::string result = text;

  // Replace various dashes and symbols
  struct Replacement {
    const char *from;
    const char *to;
  };

  const Replacement replacements[] = {
      {"–", "-"},          // en dash
      {"‑", "-"},          // non-breaking hyphen
      {"—", "-"},          // em dash
      {"_", " "},          // underscore
      {u8"\u201C", "\""},  // left double quote
      {u8"\u201D", "\""},  // right double quote
      {u8"\u2018", "'"},   // left single quote
      {u8"\u2019", "'"},   // right single quote
      {"´", "'"},          // acute accent
      {"`", "'"},          // grave accent
      {"[", " "},          // left bracket
      {"]", " "},          // right bracket
      {"|", " "},          // vertical bar
      {"/", " "},          // slash
      {"#", " "},          // hash
      {"→", " "},          // right arrow
      {"←", " "},          // left arrow
  };

  for (const auto &repl : replacements) {
    ReplaceString(&result, repl.from, repl.to);
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

  // Remove special symbols
  const char *special_symbols[] = {"♥", "☆", "♡", "©", "\\"};
  for (const char *symbol : special_symbols) {
    ReplaceString(&result, symbol, "");
  }

  // Replace known expressions
  const Replacement expr_replacements[] = {
      {"@", " at "},
      {"e.g.,", "for example, "},
      {"i.e.,", "that is, "},
  };

  for (const auto &repl : expr_replacements) {
    ReplaceString(&result, repl.from, repl.to);
  }

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

  // Add period if text doesn't end with punctuation
  if (!result.empty()) {
    char last_char = result.back();
    bool ends_with_punct =
        (last_char == '.' || last_char == '!' || last_char == '?' ||
         last_char == ';' || last_char == ':' || last_char == ',' ||
         last_char == '\'' || last_char == '"' || last_char == ')' ||
         last_char == ']' || last_char == '}' || last_char == '>');

    if (!ends_with_punct && result.size() >= 3) {
      std::string last_three = result.substr(result.size() - 3);
      if (last_three == "…" || last_three == "。" || last_three == "」" ||
          last_three == "』" || last_three == "】" || last_three == "〉" ||
          last_three == "》" || last_three == "›" || last_three == "»" ||
          last_three == u8"\u201C" || last_three == u8"\u201D" ||
          last_three == u8"\u2018" || last_three == u8"\u2019") {
        ends_with_punct = true;
      }
    }

    if (!ends_with_punct) {
      result += ".";
    }
  }

  // Validate language
  bool valid_lang = false;
  for (const auto &available_lang : kSupertonicAvailableLangs) {
    if (lang == available_lang) {
      valid_lang = true;
      break;
    }
  }
  if (!valid_lang) {
    SHERPA_ONNX_LOGE("Invalid language: %s. Available: en, ko, es, pt, fr",
                     lang.c_str());
    SHERPA_ONNX_EXIT(-1);
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

void SupertonicUnicodeProcessor::GetTextMask(
    const std::vector<int64_t> &text_ids_lengths, std::vector<float> *mask_flat,
    std::vector<int64_t> *mask_shape) const {
  int bsz = static_cast<int>(text_ids_lengths.size());
  LengthToMaskFlat(text_ids_lengths, bsz, static_cast<int64_t>(-1), mask_flat,
                   mask_shape);
}

void SupertonicUnicodeProcessor::Process(
    const std::vector<std::string> &text_list,
    const std::vector<std::string> &lang_list,
    std::vector<std::vector<int64_t>> *text_ids,
    std::vector<float> *text_mask_flat,
    std::vector<int64_t> *text_mask_shape) const {
  // Validate input sizes
  if (text_list.size() != lang_list.size()) {
    SHERPA_ONNX_LOGE(
        "Process: text_list.size() (%zu) != lang_list.size() (%zu)",
        text_list.size(), lang_list.size());
    SHERPA_ONNX_EXIT(-1);
  }

  // Handle empty batch case to avoid UB with std::max_element
  if (text_list.empty()) {
    text_ids->clear();
    GetTextMask(std::vector<int64_t>(), text_mask_flat, text_mask_shape);
    return;
  }

  std::vector<std::string> processed_texts;
  for (size_t i = 0; i < text_list.size(); i++) {
    processed_texts.push_back(PreprocessText(text_list[i], lang_list[i]));
  }

  std::vector<std::vector<uint16_t>> all_unicode_vals;
  std::vector<int64_t> text_ids_lengths;
  for (const auto &text : processed_texts) {
    auto unicode_vals = TextToUnicodeValues(text);
    text_ids_lengths.push_back(static_cast<int64_t>(unicode_vals.size()));
    all_unicode_vals.push_back(std::move(unicode_vals));
  }

  // text_ids_lengths is guaranteed to be non-empty here (same size as
  // text_list)
  int64_t max_len =
      *std::max_element(text_ids_lengths.begin(), text_ids_lengths.end());

  text_ids->resize(text_list.size());
  for (size_t i = 0; i < all_unicode_vals.size(); i++) {
    (*text_ids)[i].resize(max_len, 0);
    const auto &unicode_vals = all_unicode_vals[i];
    for (size_t j = 0; j < unicode_vals.size(); j++) {
      if (unicode_vals[j] < indexer_.size()) {
        (*text_ids)[i][j] = indexer_[unicode_vals[j]];
      }
    }
  }

  GetTextMask(text_ids_lengths, text_mask_flat, text_mask_shape);
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
