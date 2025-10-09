// sherpa-onnx/csrc/jieba-lexicon.cc
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/jieba-lexicon.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <strstream>
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

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

static bool IsPunct(const std::string &s) {
  static const std::unordered_set<std::string> puncts = {
      ",",  ".",  "!",  "?", ":", "\"", "'", "，",
      "。", "！", "？", "“", "”", "‘",  "’",
  };
  return puncts.count(s);
}

class JiebaLexicon::Impl {
 public:
  Impl(const std::string &lexicon, const std::string &tokens,
       const std::string &dict_dir, bool debug)
      : debug_(debug) {
    if (!dict_dir.empty()) {
      SHERPA_ONNX_LOGE(
          "From sherpa-onnx v1.12.15, you don't need to provide dict_dir or "
          "dictDir for this model");
      SHERPA_ONNX_LOGE("It is ignored if you provide it");
    }
    if (lexicon.empty()) {
      SHERPA_ONNX_LOGE("Please provide lexicon.txt for this model");
      SHERPA_ONNX_EXIT(-1);
    }

    {
      std::ifstream is(tokens);
      InitTokens(is);
    }

    {
      std::ifstream is(lexicon);
      InitLexicon(is);
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const std::string &lexicon, const std::string &tokens,
       const std::string &dict_dir, bool debug)
      : debug_(debug) {
    if (!dict_dir.empty()) {
      SHERPA_ONNX_LOGE(
          "From sherpa-onnx v1.12.15, you don't need to provide dict_dir or "
          "dictDir for this model");
      SHERPA_ONNX_LOGE("It is ignored if you provide it");
    }

    if (lexicon.empty()) {
      SHERPA_ONNX_LOGE("Please provide lexicon.txt for this model");
      SHERPA_ONNX_EXIT(-1);
    }

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
  }

  std::vector<TokenIDs> ConvertTextToTokenIds(const std::string &text) const {
    // see
    // https://github.com/Plachtaa/VITS-fast-fine-tuning/blob/main/text/mandarin.py#L244
    std::regex punct_re{"：|、|；"};
    std::string s = std::regex_replace(text, punct_re, "，");

    std::regex punct_re2("[.]");
    s = std::regex_replace(s, punct_re2, "。");

    std::regex punct_re3("[?]");
    s = std::regex_replace(s, punct_re3, "？");

    std::regex punct_re4("[!]");
    s = std::regex_replace(s, punct_re4, "！");

    std::vector<std::string> words = SplitUtf8(text);

    if (debug_) {
#if __OHOS__
      SHERPA_ONNX_LOGE("input text:\n%{public}s", text.c_str());
      SHERPA_ONNX_LOGE("after replacing punctuations:\n%{public}s", s.c_str());
#else
      SHERPA_ONNX_LOGE("input text:\n%s", text.c_str());
      SHERPA_ONNX_LOGE("after replacing punctuations:\n%s", s.c_str());
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

    int32_t num_words = static_cast<int32_t>(words.size());
    int32_t max_search_len = 10;

    for (int32_t i = 0; i < num_words;) {
      int32_t start = i;
      int32_t end = std::min(i + max_search_len, num_words - 1);

      std::string w;
      while (end > start) {
        auto this_word = GetWord(words, start, end);
        if (debug_) {
#if __OHOS__
          SHERPA_ONNX_LOGE("%{public}d-%{public}d: %{public}s", start, end,
                           this_word.c_str());
#else
          SHERPA_ONNX_LOGE("%d-%d: %s", start, end, this_word.c_str());
#endif
        }
        if (word2ids_.count(this_word)) {
          i = end + 1;
          w = std::move(this_word);
          if (debug_) {
#if __OHOS__
            SHERPA_ONNX_LOGE("matched %{public}d-%{public}d: %{public}s", start,
                             end, w.c_str());
#else
            SHERPA_ONNX_LOGE("matched %d-%d: %s", start, end, w.c_str());
#endif
          }
          break;
        }

        end -= 1;
      }

      if (w.empty()) {
        w = words[i];
        i += 1;
      }

      auto ids = ConvertWordToIds(w);
      if (ids.empty()) {
#if __OHOS__
        SHERPA_ONNX_LOGE("Ignore OOV '%{public}s'", w.c_str());
#else
        SHERPA_ONNX_LOGE("Ignore OOV '%s'", w.c_str());
#endif
        continue;
      }

      this_sentence.insert(this_sentence.end(), ids.begin(), ids.end());

      if (IsPunct(w)) {
        ans.emplace_back(std::move(this_sentence));
        this_sentence = {};
      }
    }  // for (int32_t i = 0; i < num_words;)

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
      std::vector<std::string> words = SplitUtf8(w);
      for (const auto &word : words) {
        if (word2ids_.count(word)) {
          auto ids = ConvertWordToIds(word);
          ans.insert(ans.end(), ids.begin(), ids.end());
        }
      }
    }
    if (debug_) {
      std::ostringstream os;
      os << w << ": ";
      for (auto i : ans) {
        os << id2token_.at(i) << " ";
      }
      os << "\n";
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

    std::vector<std::pair<std::string, std::string>> puncts = {
        {",", "，"}, {".", "。"}, {"!", "！"}, {"?", "？"}, {":", "："},
        {"\"", "“"}, {"\"", "”"}, {"'", "‘"},  {"'", "’"},  {";", "；"},
    };

    for (const auto &p : puncts) {
      if (token2id_.count(p.first) && !token2id_.count(p.second)) {
        token2id_[p.second] = token2id_[p.first];
      }

      if (!token2id_.count(p.first) && token2id_.count(p.second)) {
        token2id_[p.first] = token2id_[p.second];
      }
    }

    if (!token2id_.count("、") && token2id_.count("，")) {
      token2id_["、"] = token2id_["，"];
    }

    if (!token2id_.count(";") && token2id_.count(",")) {
      token2id_[";"] = token2id_[","];
    }

    if (debug_) {
      for (const auto &p : token2id_) {
        id2token_[p.second] = p.first;
      }
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
  }

 private:
  // lexicon.txt is saved in word2ids_
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;

  // tokens.txt is saved in token2id_
  std::unordered_map<std::string, int32_t> token2id_;

  std::unordered_map<int32_t, std::string> id2token_;

  bool debug_ = false;
};

JiebaLexicon::~JiebaLexicon() = default;

JiebaLexicon::JiebaLexicon(const std::string &lexicon,
                           const std::string &tokens,
                           const std::string &dict_dir, bool debug)
    : impl_(std::make_unique<Impl>(lexicon, tokens, dict_dir, debug)) {}

template <typename Manager>
JiebaLexicon::JiebaLexicon(Manager *mgr, const std::string &lexicon,
                           const std::string &tokens,
                           const std::string &dict_dir, bool debug)
    : impl_(std::make_unique<Impl>(mgr, lexicon, tokens, dict_dir, debug)) {}

std::vector<TokenIDs> JiebaLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string & /*unused_voice = ""*/) const {
  return impl_->ConvertTextToTokenIds(text);
}

#if __ANDROID_API__ >= 9
template JiebaLexicon::JiebaLexicon(AAssetManager *mgr,
                                    const std::string &lexicon,
                                    const std::string &tokens,
                                    const std::string &dict_dir, bool debug);
#endif

#if __OHOS__
template JiebaLexicon::JiebaLexicon(NativeResourceManager *mgr,
                                    const std::string &lexicon,
                                    const std::string &tokens,
                                    const std::string &dict_dir, bool debug);
#endif

}  // namespace sherpa_onnx
