// sherpa-onnx/csrc/lexicon.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/lexicon.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <utility>

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include <memory>
#include <regex>  // NOLINT

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

static void ToLowerCase(std::string *in_out) {
  std::transform(in_out->begin(), in_out->end(), in_out->begin(),
                 [](unsigned char c) { return std::tolower(c); });
}

// Note: We don't use SymbolTable here since tokens may contain a blank
// in the first column
static std::unordered_map<std::string, int32_t> ReadTokens(std::istream &is) {
  std::unordered_map<std::string, int32_t> token2id;

  std::string line;

  std::string sym;
  int32_t id;
  while (std::getline(is, line)) {
    std::istringstream iss(line);
    iss >> sym;
    if (iss.eof()) {
      id = atoi(sym.c_str());
      sym = " ";
    } else {
      iss >> id;
    }

    // eat the trailing \r\n on windows
    iss >> std::ws;
    if (!iss.eof()) {
      SHERPA_ONNX_LOGE("Error: %s", line.c_str());
      exit(-1);
    }

#if 0
    if (token2id.count(sym)) {
      SHERPA_ONNX_LOGE("Duplicated token %s. Line %s. Existing ID: %d",
                       sym.c_str(), line.c_str(), token2id.at(sym));
      exit(-1);
    }
#endif
    token2id.insert({std::move(sym), id});
  }

  return token2id;
}

static std::vector<int32_t> ConvertTokensToIds(
    const std::unordered_map<std::string, int32_t> &token2id,
    const std::vector<std::string> &tokens) {
  std::vector<int32_t> ids;
  ids.reserve(tokens.size());
  for (const auto &s : tokens) {
    if (!token2id.count(s)) {
      return {};
    }
    int32_t id = token2id.at(s);
    ids.push_back(id);
  }

  return ids;
}

Lexicon::Lexicon(const std::string &lexicon, const std::string &tokens,
                 const std::string &punctuations, const std::string &language,
                 bool debug /*= false*/, bool is_piper /*= false*/)
    : debug_(debug), is_piper_(is_piper) {
  InitLanguage(language);

  {
    std::ifstream is(tokens);
    InitTokens(is);
  }

  {
    std::ifstream is(lexicon);
    InitLexicon(is);
  }

  InitPunctuations(punctuations);
}

#if __ANDROID_API__ >= 9
Lexicon::Lexicon(AAssetManager *mgr, const std::string &lexicon,
                 const std::string &tokens, const std::string &punctuations,
                 const std::string &language, bool debug /*= false*/,
                 bool is_piper /*= false*/)
    : debug_(debug), is_piper_(is_piper) {
  InitLanguage(language);

  {
    auto buf = ReadFile(mgr, tokens);
    std::istrstream is(buf.data(), buf.size());
    InitTokens(is);
  }

  {
    auto buf = ReadFile(mgr, lexicon);
    std::istrstream is(buf.data(), buf.size());
    InitLexicon(is);
  }

  InitPunctuations(punctuations);
}
#endif

std::vector<std::vector<int64_t>> Lexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string & /*voice*/ /*= ""*/) const {
  switch (language_) {
    case Language::kEnglish:
      return ConvertTextToTokenIdsEnglish(text);
    case Language::kGerman:
      return ConvertTextToTokenIdsGerman(text);
    case Language::kSpanish:
      return ConvertTextToTokenIdsSpanish(text);
    case Language::kFrench:
      return ConvertTextToTokenIdsFrench(text);
    case Language::kChinese:
      return ConvertTextToTokenIdsChinese(text);
    default:
      SHERPA_ONNX_LOGE("Unknown language: %d", static_cast<int32_t>(language_));
      exit(-1);
  }

  return {};
}

std::vector<std::vector<int64_t>> Lexicon::ConvertTextToTokenIdsChinese(
    const std::string &text) const {
  std::vector<std::string> words;
  if (pattern_) {
    // Handle polyphones
    size_t pos = 0;
    auto begin = std::sregex_iterator(text.begin(), text.end(), *pattern_);
    auto end = std::sregex_iterator();
    for (std::sregex_iterator i = begin; i != end; ++i) {
      std::smatch match = *i;
      if (pos < match.position()) {
        auto this_segment = text.substr(pos, match.position() - pos);
        auto this_segment_words = SplitUtf8(this_segment);
        words.insert(words.end(), this_segment_words.begin(),
                     this_segment_words.end());
        pos = match.position() + match.length();
      } else if (pos == match.position()) {
        pos = match.position() + match.length();
      }

      words.push_back(match.str());
    }

    if (pos < text.size()) {
      auto this_segment = text.substr(pos, text.size() - pos);
      auto this_segment_words = SplitUtf8(this_segment);
      words.insert(words.end(), this_segment_words.begin(),
                   this_segment_words.end());
    }
  } else {
    words = SplitUtf8(text);
  }

  if (debug_) {
    fprintf(stderr, "Input text in string: %s\n", text.c_str());
    fprintf(stderr, "Input text in bytes:");
    for (uint8_t c : text) {
      fprintf(stderr, " %02x", c);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "After splitting to words:");
    for (const auto &w : words) {
      fprintf(stderr, " %s", w.c_str());
    }
    fprintf(stderr, "\n");
  }

  std::vector<int64_t> ans;

  int32_t blank = -1;
  if (token2id_.count(" ")) {
    blank = token2id_.at(" ");
  }

  int32_t sil = -1;
  int32_t eos = -1;
  if (token2id_.count("sil")) {
    sil = token2id_.at("sil");
    eos = token2id_.at("eos");
  }

  if (sil != -1) {
    ans.push_back(sil);
  }

  for (const auto &w : words) {
    if (punctuations_.count(w)) {
      if (token2id_.count(w)) {
        ans.push_back(token2id_.at(w));
      } else if (sil != -1) {
        ans.push_back(sil);
      }
      continue;
    }

    if (!word2ids_.count(w)) {
      SHERPA_ONNX_LOGE("OOV %s. Ignore it!", w.c_str());
      continue;
    }

    const auto &token_ids = word2ids_.at(w);
    ans.insert(ans.end(), token_ids.begin(), token_ids.end());
    if (blank != -1) {
      ans.push_back(blank);
    }
  }

  if (sil != -1) {
    ans.push_back(sil);
  }

  if (eos != -1) {
    ans.push_back(eos);
  }

  return {ans};
}

std::vector<std::vector<int64_t>> Lexicon::ConvertTextToTokenIdsEnglish(
    const std::string &_text) const {
  std::string text(_text);
  ToLowerCase(&text);

  std::vector<std::string> words = SplitUtf8(text);

  if (debug_) {
    fprintf(stderr, "Input text (lowercase) in string: %s\n", text.c_str());
    fprintf(stderr, "Input text in bytes:");
    for (uint8_t c : text) {
      fprintf(stderr, " %02x", c);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "After splitting to words:");
    for (const auto &w : words) {
      fprintf(stderr, " %s", w.c_str());
    }
    fprintf(stderr, "\n");
  }

  int32_t blank = token2id_.at(" ");

  std::vector<int64_t> ans;
  if (is_piper_ && token2id_.count("^")) {
    ans.push_back(token2id_.at("^"));  // sos
  }

  for (const auto &w : words) {
    if (punctuations_.count(w)) {
      ans.push_back(token2id_.at(w));
      continue;
    }

    if (!word2ids_.count(w)) {
      SHERPA_ONNX_LOGE("OOV %s. Ignore it!", w.c_str());
      continue;
    }

    const auto &token_ids = word2ids_.at(w);
    ans.insert(ans.end(), token_ids.begin(), token_ids.end());
    ans.push_back(blank);
  }

  if (!ans.empty()) {
    // remove the last blank
    ans.resize(ans.size() - 1);
  }

  if (is_piper_ && token2id_.count("$")) {
    ans.push_back(token2id_.at("$"));  // eos
  }

  return {ans};
}

void Lexicon::InitTokens(std::istream &is) { token2id_ = ReadTokens(is); }

void Lexicon::InitLanguage(const std::string &_lang) {
  std::string lang(_lang);
  ToLowerCase(&lang);
  if (lang == "english") {
    language_ = Language::kEnglish;
  } else if (lang == "german") {
    language_ = Language::kGerman;
  } else if (lang == "spanish") {
    language_ = Language::kSpanish;
  } else if (lang == "french") {
    language_ = Language::kFrench;
  } else if (lang == "chinese") {
    language_ = Language::kChinese;
  } else {
    SHERPA_ONNX_LOGE("Unknown language: %s", _lang.c_str());
    exit(-1);
  }
}

void Lexicon::InitLexicon(std::istream &is) {
  std::string word;
  std::vector<std::string> token_list;
  std::string line;
  std::string phone;

  std::ostringstream os;
  std::string sep;

  while (std::getline(is, line)) {
    std::istringstream iss(line);

    token_list.clear();

    iss >> word;
    ToLowerCase(&word);

    if (word2ids_.count(word)) {
      SHERPA_ONNX_LOGE("Duplicated word: %s. Ignore it.", word.c_str());
      continue;
    }

    while (iss >> phone) {
      token_list.push_back(std::move(phone));
    }

    std::vector<int32_t> ids = ConvertTokensToIds(token2id_, token_list);
    if (ids.empty()) {
      continue;
    }
    if (language_ == Language::kChinese && word.size() > 3) {
      // this is not a single word;
      os << sep << word;
      sep = "|";
    }

    word2ids_.insert({std::move(word), std::move(ids)});
  }

  if (!sep.empty()) {
    pattern_ = std::make_unique<std::regex>(os.str());
  }
}

void Lexicon::InitPunctuations(const std::string &punctuations) {
  std::vector<std::string> punctuation_list;
  SplitStringToVector(punctuations, " ", false, &punctuation_list);
  for (auto &s : punctuation_list) {
    punctuations_.insert(std::move(s));
  }
}

}  // namespace sherpa_onnx
