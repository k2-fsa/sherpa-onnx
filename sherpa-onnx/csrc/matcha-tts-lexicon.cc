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

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/phrase-matcher.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class MatchaTtsLexicon::Impl {
 public:
  Impl(const std::string &lexicon, const std::string &tokens,
       const std::string & /*data_dir*/, bool debug,
       bool /*skip_replacement*/)
      : debug_(debug) {
    if (lexicon.empty()) {
      SHERPA_ONNX_LOGE("Please provide lexicon.txt for this model");
      SHERPA_ONNX_EXIT(-1);
    }

    {
      auto is = OpenInputFile(tokens);
      InitTokens(is);
    }

    InitLexicon(lexicon);

    // data_dir (the eSpeak-ng phoneme-data directory) has no purpose in this
    // build; the parameter is kept for config compatibility and ignored.
  }

  template <typename Manager>
  Impl(Manager *mgr, const std::string &lexicon, const std::string &tokens,
       const std::string & /*data_dir*/, bool debug,
       bool /*skip_replacement*/)
      : debug_(debug) {
    if (lexicon.empty()) {
      SHERPA_ONNX_LOGE("Please provide lexicon.txt for this model");
      SHERPA_ONNX_EXIT(-1);
    }

    {
      auto buf = ReadFile(mgr, tokens);
      std::istringstream is(std::string(buf.data(), buf.size()));

      InitTokens(is);
    }

    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      auto buf = ReadFile(mgr, f);

      std::istringstream is(std::string(buf.data(), buf.size()));
      InitLexicon(is);
    }

    // data_dir has no purpose in this build; ignored (see above).
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
        // The eSpeak-based phonemization engine that upstream used for
        // out-of-lexicon words has been removed from this build, so such a
        // word cannot be converted to tokens; drop it and warn once.
        OfflineTtsLogPhonemizationRemovedOnce();
        if (debug_) {
#if __OHOS__
          SHERPA_ONNX_LOGE("Drop OOV word not in lexicon: %{public}s", w.c_str());
#else
          SHERPA_ONNX_LOGE("Drop OOV word not in lexicon: %s", w.c_str());
#endif
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
      auto is = OpenInputFile(f);
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
                                   const std::string &data_dir, bool debug,
                                   bool skip_replacement)
    : impl_(std::make_unique<Impl>(lexicon, tokens, data_dir, debug,
                                   skip_replacement)) {}  // NOLINT

template <typename Manager>
MatchaTtsLexicon::MatchaTtsLexicon(Manager *mgr, const std::string &lexicon,
                                   const std::string &tokens,
                                   const std::string &data_dir, bool debug,
                                   bool skip_replacement)
    : impl_(std::make_unique<Impl>(mgr, lexicon, tokens, data_dir, debug,
                                   skip_replacement)) {}  // NOLINT

std::vector<TokenIDs> MatchaTtsLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string & /*unused_voice = ""*/) const {
  return impl_->ConvertTextToTokenIds(text);
}

#if __ANDROID_API__ >= 9
template MatchaTtsLexicon::MatchaTtsLexicon(AAssetManager *mgr,
                                            const std::string &lexicon,
                                            const std::string &tokens,
                                            const std::string &data_dir,
                                            bool debug, bool skip_replacement);
#endif

#if __OHOS__
template MatchaTtsLexicon::MatchaTtsLexicon(NativeResourceManager *mgr,
                                            const std::string &lexicon,
                                            const std::string &tokens,
                                            const std::string &data_dir,
                                            bool debug, bool skip_replacement);
#endif

}  // namespace sherpa_onnx
