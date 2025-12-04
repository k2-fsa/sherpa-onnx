// sherpa-onnx/csrc/matcha-tts-lexicon.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/matcha-tts-lexicon.h"

#include <ctype.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <strstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "espeak-ng/speak_lib.h"
#include "phoneme_ids.hpp"  // NOLINT
#include "phonemize.hpp"    // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/phrase-matcher.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {
// code in this anonymous namespace is written by ChatGPT
//
// Please see https://github.com/k2-fsa/sherpa-onnx/pull/2853
// for why we need to do the replacement
static const std::vector<std::pair<std::string, std::string>> kReplacements = {
    {"ɝ", "ɜɹ"}, {"ɚ", "əɹ"},

    {"eɪ", "A"}, {"aɪ", "I"}, {"ɔɪ", "Y"},
    {"oʊ", "O"}, {"əʊ", "O"}, {"aʊ", "W"},

    {"tʃ", "ʧ"}, {"dʒ", "ʤ"},

    {"ː", ""},

    {"g", "ɡ"},  {"r", "ɹ"},

    {"e", "ɛ"},
};

std::string Utf32ToUtf8(char32_t cp) {
  std::string out;

  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }

  return out;
}

std::vector<std::string> ConvertPhonemesToUTF8(
    const std::vector<std::vector<char32_t>> &phonemes) {
  std::vector<std::string> out;

  for (const auto &word : phonemes) {
    for (char32_t cp : word) {
      out.push_back(Utf32ToUtf8(cp));
    }
  }

  return out;
}

std::string ApplyReplacements(std::string s) {
  for (const auto &p : kReplacements) {
    const std::string &from = p.first;
    const std::string &to = p.second;

    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      s.replace(pos, from.size(), to);
      pos += to.size();
    }
  }
  return s;
}

std::vector<std::string> SplitTokensUTF8(const std::string &s) {
  std::vector<std::string> out;

  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    size_t len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;

    out.push_back(s.substr(i, len));
    i += len;
  }

  return out;
}

std::vector<std::string> ProcessPhonemes(
    const std::vector<std::vector<char32_t>> &phonemes) {
  auto tokens = ConvertPhonemesToUTF8(phonemes);
  std::string joined = Join(tokens);
  std::string replaced = ApplyReplacements(joined);
  return SplitTokensUTF8(replaced);
}

}  // namespace

void CallPhonemizeEspeak(const std::string &text,
                         piper::eSpeakPhonemeConfig &config,  // NOLINT
                         std::vector<std::vector<piper::Phoneme>> *phonemes);

class MatchaTtsLexicon::Impl {
 public:
  Impl(const std::string &lexicon, const std::string &tokens,
       const std::string &data_dir, bool debug)
      : debug_(debug) {
    if (lexicon.empty()) {
      SHERPA_ONNX_LOGE("Please provide lexicon.txt for this model");
      SHERPA_ONNX_EXIT(-1);
    }

    {
      std::ifstream is(tokens);
      InitTokens(is);
    }

    InitLexicon(lexicon);

    if (data_dir.empty()) {
      SHERPA_ONNX_LOGE("Please provide data dir for this model");
      SHERPA_ONNX_EXIT(-1);
    }

    InitEspeak(data_dir);  // See ./piper-phonemize-lexicon.cc
  }

  template <typename Manager>
  Impl(Manager *mgr, const std::string &lexicon, const std::string &tokens,
       const std::string &data_dir, bool debug)
      : debug_(debug) {
    if (lexicon.empty()) {
      SHERPA_ONNX_LOGE("Please provide lexicon.txt for this model");
      SHERPA_ONNX_EXIT(-1);
    }

    {
      auto buf = ReadFile(mgr, tokens);
      std::istrstream is(buf.data(), buf.size());

      InitTokens(is);
    }

    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      auto buf = ReadFile(mgr, f);

      std::istrstream is(buf.data(), buf.size());
      InitLexicon(is);
    }

    if (data_dir.empty()) {
      SHERPA_ONNX_LOGE("Please provide data dir for this model");
      SHERPA_ONNX_EXIT(-1);
    }

    InitEspeak(data_dir);  // See ./piper-phonemize-lexicon.cc
  }

  std::vector<TokenIDs> ConvertTextToTokenIds(const std::string &_text) const {
    std::string text = _text;
    std::vector<std::pair<std::string, std::string>> replace_str_pairs = {
        {"，", ","}, {"、", ","}, {"；", ";"}, {"：", ","},   {":", ","},
        {"。", "."}, {"？", "?"}, {"！", "!"}, {"\\s+", " "},
    };
    for (const auto &p : replace_str_pairs) {
      std::regex re(p.first);
      text = std::regex_replace(text, re, p.second);
    }

    if (debug_) {
      SHERPA_ONNX_LOGE("After replacing punctuations and merging spaces:\n%s",
                       text.c_str());
    }

    std::vector<std::string> words = SplitUtf8(text);

    if (debug_) {
#if __OHOS__
      SHERPA_ONNX_LOGE("input text:\n%{public}s", _text.c_str());
      SHERPA_ONNX_LOGE("after replacing punctuations:\n%{public}s",
                       text.c_str());
#else
      SHERPA_ONNX_LOGE("input text:\n%s", _text.c_str());
      SHERPA_ONNX_LOGE("after replacing punctuations:\n%s", text.c_str());
#endif

      std::ostringstream os;
      std::string sep = "";
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("after splitting into UTF8:\n%{public}s",
                       os.str().c_str());
#else
      SHERPA_ONNX_LOGE("after splitting into UTF8:\n%s", os.str().c_str());
#endif
    }

    // remove spaces after punctuations
    std::vector<std::string> words2 = std::move(words);
    words.reserve(words2.size());

    for (int32_t i = 0; i < words2.size(); ++i) {
      if (i == 0) {
        words.push_back(std::move(words2[i]));
      } else if (words2[i] == " ") {
        if (words.back() == " " || IsPunct(words.back())) {
          continue;
        } else {
          words.push_back(std::move(words2[i]));
        }
      } else if (IsPunct(words2[i])) {
        if (words.back() == " " || IsPunct(words.back())) {
          continue;
        } else {
          words.push_back(std::move(words2[i]));
        }
      } else {
        words.push_back(std::move(words2[i]));
      }
    }

    if (debug_) {
      std::ostringstream os;
      std::string sep = "";
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("after removing spaces after punctuations:\n%{public}s",
                       os.str().c_str());
#else
      SHERPA_ONNX_LOGE("after removing spaces after punctuations:\n%s",
                       os.str().c_str());
#endif
    }

    std::vector<TokenIDs> ans;
    std::vector<int64_t> this_sentence;

    PhraseMatcher matcher(&all_words_, words, debug_);

    int32_t blank = token2id_.at(" ");

    std::vector<int32_t> ids;
    std::string last_word;
    for (const std::string &w : matcher) {
      ids = ConvertWordToIds(w);

      if (ids.empty()) {
#if __OHOS__
        SHERPA_ONNX_LOGE("Ignore OOV '%{public}s'", w.c_str());
#else
        SHERPA_ONNX_LOGE("Ignore OOV '%s'", w.c_str());
#endif

        last_word = w;
        continue;
      }

      if (!last_word.empty() && isalpha(last_word[0])) {
        this_sentence.push_back(blank);
      }

      this_sentence.insert(this_sentence.end(), ids.begin(), ids.end());

      if (IsPunct(w)) {
        if (debug_) {
          std::ostringstream os;
          std::string sep;
          os << "new sentence: [";
          for (auto i : this_sentence) {
            os << sep << i;
            sep = ", ";
          }
          os << "]";
          SHERPA_ONNX_LOGE("%s", os.str().c_str());
        }

        ans.emplace_back(std::move(this_sentence));
        this_sentence = {};
      }

      last_word = w;
    }  // for (const std::string &w : matcher)

    if (!this_sentence.empty()) {
      ans.emplace_back(std::move(this_sentence));
    }

    return ans;
  }

 private:
  std::vector<int32_t> ConvertWordToIds(const std::string &w) const {
    std::vector<int32_t> ans;
    if (word2ids_.count(w)) {
      ans = word2ids_.at(w);
    } else if (token2id_.count(w)) {
      ans = {token2id_.at(w)};
    } else {
      if (ContainsCJK(w)) {
        std::vector<std::string> words = SplitUtf8(w);
        for (const auto &word : words) {
          if (word2ids_.count(word)) {
            auto ids = ConvertWordToIds(word);
            ans.insert(ans.end(), ids.begin(), ids.end());
          }
        }
      } else {
        if (debug_) {
          SHERPA_ONNX_LOGE("use espeak for %s", w.c_str());
        }
        // use espeak
        piper::eSpeakPhonemeConfig config;
        config.voice = "en-us";
        std::vector<std::vector<piper::Phoneme>> phonemes;
        CallPhonemizeEspeak(w, config, &phonemes);

        auto pp = ProcessPhonemes(phonemes);

        for (const auto &p : pp) {
          if (token2id_.count(p)) {
            ans.push_back(token2id_.at(p));
          } else {
            SHERPA_ONNX_LOGE("Skip token: %s", p.c_str());
          }
        }
      }
    }

    if (debug_) {
      std::ostringstream os;
      os << w << ": ";
      for (auto i : ans) {
        os << "'" << id2token_.at(i) << "'(" << i << ")" << ",";
      }
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

    return ans;
  }

  void InitTokens(std::istream &is) {
    token2id_ = ReadTokens(is);

    if (debug_) {
      for (const auto &p : token2id_) {
        id2token_[p.second] = p.first;
      }
    }
  }

  void InitLexicon(const std::string &lexicon) {
    if (lexicon.empty()) {
      SHERPA_ONNX_LOGE("Empty lexicon!");
      return;
    }

    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      std::ifstream is(f);
      InitLexicon(is);
    }
  }

  void InitLexicon(std::istream &is) {
    std::string word;
    std::vector<std::string> token_list;
    std::string line;
    std::string phone;
    int32_t line_num = 0;

    while (std::getline(is, line)) {
      ++line_num;

      std::istringstream iss(line);

      token_list.clear();

      iss >> word;
      ToLowerCase(&word);

      if (word2ids_.count(word)) {
#if __OHOS__
        SHERPA_ONNX_LOGE(
            "Duplicated word: %{public}s at line %{public}d:%{public}s. Ignore "
            "it.",
            word.c_str(), line_num, line.c_str());
#else
        SHERPA_ONNX_LOGE("Duplicated word: %s at line %d:%s. Ignore it.",
                         word.c_str(), line_num, line.c_str());
#endif
        continue;
      }

      while (iss >> phone) {
        token_list.push_back(std::move(phone));
      }

      std::vector<int32_t> ids = ConvertTokensToIds(token2id_, token_list);
      if (ids.empty()) {
        if (debug_) {
#if __OHOS__
          SHERPA_ONNX_LOGE("Empty token ids for '%{public}s'", line.c_str());
#else
          SHERPA_ONNX_LOGE("Empty token ids for '%s'", line.c_str());
#endif
        }
        continue;
      }

      word2ids_.insert({std::move(word), std::move(ids)});
    }

    for (const auto &[key, _] : word2ids_) {
      all_words_.insert(key);
    }
  }

 private:
  // lexicon.txt is saved in word2ids_
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;
  std::unordered_set<std::string> all_words_;

  // tokens.txt is saved in token2id_
  std::unordered_map<std::string, int32_t> token2id_;

  std::unordered_map<int32_t, std::string> id2token_;

  bool debug_ = false;
};  // namespace sherpa_onnx

MatchaTtsLexicon::~MatchaTtsLexicon() = default;

MatchaTtsLexicon::MatchaTtsLexicon(const std::string &lexicon,
                                   const std::string &tokens,
                                   const std::string &data_dir, bool debug)
    : impl_(std::make_unique<Impl>(lexicon, tokens, data_dir, debug)) {}

template <typename Manager>
MatchaTtsLexicon::MatchaTtsLexicon(Manager *mgr, const std::string &lexicon,
                                   const std::string &tokens,
                                   const std::string &data_dir, bool debug)
    : impl_(std::make_unique<Impl>(mgr, lexicon, tokens, data_dir, debug)) {}

std::vector<TokenIDs> MatchaTtsLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string & /*unused_voice = ""*/) const {
  return impl_->ConvertTextToTokenIds(text);
}

#if __ANDROID_API__ >= 9
template MatchaTtsLexicon::MatchaTtsLexicon(AAssetManager *mgr,
                                            const std::string &lexicon,
                                            const std::string &tokens,
                                            const std::string &data_dir,
                                            bool debug);
#endif

#if __OHOS__
template MatchaTtsLexicon::MatchaTtsLexicon(NativeResourceManager *mgr,
                                            const std::string &lexicon,
                                            const std::string &tokens,
                                            const std::string &data_dir,
                                            bool debug);
#endif

}  // namespace sherpa_onnx
