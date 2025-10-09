// sherpa-onnx/csrc/melo-tts-lexicon.cc
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/melo-tts-lexicon.h"

#include <fstream>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <strstream>
#include <unordered_map>
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

class MeloTtsLexicon::Impl {
 public:
  Impl(const std::string &lexicon, const std::string &tokens,
       const std::string &dict_dir,
       const OfflineTtsVitsModelMetaData &meta_data, bool debug)
      : meta_data_(meta_data), debug_(debug) {
    if (!dict_dir.empty()) {
      SHERPA_ONNX_LOGE(
          "From sherpa-onnx v1.12.15, you don't need to provide dict_dir or "
          "dictDir for this model");
      SHERPA_ONNX_LOGE("It is ignored if you provide it");
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

  Impl(const std::string &lexicon, const std::string &tokens,
       const OfflineTtsVitsModelMetaData &meta_data, bool debug)
      : meta_data_(meta_data), debug_(debug) {
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
       const std::string &dict_dir,
       const OfflineTtsVitsModelMetaData &meta_data, bool debug)
      : meta_data_(meta_data), debug_(debug) {
    if (!dict_dir.empty()) {
      SHERPA_ONNX_LOGE(
          "From sherpa-onnx v1.12.15, you don't need to provide dict_dir or "
          "dictDir for this model");
      SHERPA_ONNX_LOGE("It is ignored if you provide it");
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

  template <typename Manager>
  Impl(Manager *mgr, const std::string &lexicon, const std::string &tokens,
       const OfflineTtsVitsModelMetaData &meta_data, bool debug)
      : meta_data_(meta_data), debug_(debug) {
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

  std::vector<TokenIDs> ConvertTextToTokenIds(const std::string &_text) const {
    std::string text = ToLowerCase(_text);
    // see
    // https://github.com/Plachtaa/VITS-fast-fine-tuning/blob/main/text/mandarin.py#L244
    std::regex punct_re{"：|、|；"};
    std::string s = std::regex_replace(text, punct_re, ",");

    std::regex punct_re2("。");
    s = std::regex_replace(s, punct_re2, ".");

    std::regex punct_re3("？");
    s = std::regex_replace(s, punct_re3, "?");

    std::regex punct_re4("！");
    s = std::regex_replace(s, punct_re4, "!");

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

    std::vector<TokenIDs> ans;
    TokenIDs this_sentence;

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
      if (ids.tokens.empty()) {
#if __OHOS__
        SHERPA_ONNX_LOGE("Ignore OOV '%{public}s'", w.c_str());
#else
        SHERPA_ONNX_LOGE("Ignore OOV '%s'", w.c_str());
#endif
        continue;
      }

      if (debug_) {
        std::ostringstream os;
        os << w << ": ";
        for (auto i : ids.tokens) {
          os << id2token_.at(i) << " ";
        }

        for (auto i : ids.tones) {
          os << i << " ";
        }
        os << "\n";
#if __OHOS__
        SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
        SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
      }

      this_sentence.tokens.insert(this_sentence.tokens.end(),
                                  ids.tokens.begin(), ids.tokens.end());
      this_sentence.tones.insert(this_sentence.tones.end(), ids.tones.begin(),
                                 ids.tones.end());

      if (w == "." || w == "!" || w == "?" || w == "," || w == "。" ||
          w == "！" || w == "？" || w == "，") {
        ans.push_back(std::move(this_sentence));
        this_sentence = {};
      }
    }  // for (int32_t i = 0; i < num_words;)

    if (!this_sentence.tokens.empty()) {
      ans.push_back(std::move(this_sentence));
    }

    return ans;
  }

 private:
  TokenIDs ConvertWordToIds(const std::string &w) const {
    if (word2ids_.count(w)) {
      return word2ids_.at(w);
    }

    if (token2id_.count(w)) {
      return {{token2id_.at(w)}, {0}};
    }

    TokenIDs ans;

    std::vector<std::string> words = SplitUtf8(w);
    for (const auto &word : words) {
      if (word2ids_.count(word)) {
        auto ids = ConvertWordToIds(word);
        ans.tokens.insert(ans.tokens.end(), ids.tokens.begin(),
                          ids.tokens.end());
        ans.tones.insert(ans.tones.end(), ids.tones.begin(), ids.tones.end());
      } else {
        // If the lexicon does not contain the word, we split the word into
        // characters.
        //
        // For instance, if the word is TTS and it is does not exist
        // in the lexicon, we split it into 3 characters: T T S
        std::string s;
        for (char c : word) {
          s = c;
          if (word2ids_.count(s)) {
            const auto &t = word2ids_.at(s);
            ans.tokens.insert(ans.tokens.end(), t.tokens.begin(),
                              t.tokens.end());
            ans.tones.insert(ans.tones.end(), t.tones.begin(), t.tones.end());
          }
        }
      }
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

    token2id_[" "] = token2id_["_"];

    std::vector<std::pair<std::string, std::string>> puncts = {
        {",", "，"}, {".", "。"}, {"!", "！"}, {"?", "？"}};

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
  }

  void InitLexicon(std::istream &is) {
    std::string word;
    std::vector<std::string> token_list;

    std::vector<std::string> phone_list;
    std::vector<int64_t> tone_list;

    std::string line;
    std::string phone;
    int32_t line_num = 0;

    while (std::getline(is, line)) {
      ++line_num;

      std::istringstream iss(line);

      token_list.clear();
      phone_list.clear();
      tone_list.clear();

      iss >> word;
      ToLowerCase(&word);

      if (word2ids_.count(word)) {
        SHERPA_ONNX_LOGE("Duplicated word: %s at line %d:%s. Ignore it.",
                         word.c_str(), line_num, line.c_str());
        continue;
      }

      while (iss >> phone) {
        token_list.push_back(std::move(phone));
      }

      if ((token_list.size() & 1) != 0) {
        SHERPA_ONNX_LOGE("Invalid line %d: '%s'", line_num, line.c_str());
        exit(-1);
      }

      int32_t num_phones = token_list.size() / 2;
      phone_list.reserve(num_phones);
      tone_list.reserve(num_phones);

      for (int32_t i = 0; i != num_phones; ++i) {
        phone_list.push_back(std::move(token_list[i]));
        tone_list.push_back(std::stoi(token_list[i + num_phones], nullptr));
        if (tone_list.back() < 0 || tone_list.back() > 50) {
          SHERPA_ONNX_LOGE("Invalid line %d: '%s'", line_num, line.c_str());
          exit(-1);
        }
      }

      std::vector<int32_t> ids = ConvertTokensToIds(token2id_, phone_list);
      if (ids.empty()) {
        continue;
      }

      if (ids.size() != num_phones) {
        SHERPA_ONNX_LOGE("Invalid line %d: '%s'", line_num, line.c_str());
        exit(-1);
      }

      std::vector<int64_t> ids64{ids.begin(), ids.end()};

      word2ids_.insert(
          {std::move(word), TokenIDs{std::move(ids64), std::move(tone_list)}});
    }

    // For Chinese+English MeloTTS
    word2ids_["呣"] = word2ids_["母"];
    word2ids_["嗯"] = word2ids_["恩"];
  }

 private:
  // lexicon.txt is saved in word2ids_
  std::unordered_map<std::string, TokenIDs> word2ids_;

  // tokens.txt is saved in token2id_
  std::unordered_map<std::string, int32_t> token2id_;
  std::unordered_map<int32_t, std::string> id2token_;

  OfflineTtsVitsModelMetaData meta_data_;

  bool debug_ = false;
};

MeloTtsLexicon::~MeloTtsLexicon() = default;

MeloTtsLexicon::MeloTtsLexicon(const std::string &lexicon,
                               const std::string &tokens,
                               const std::string &dict_dir,
                               const OfflineTtsVitsModelMetaData &meta_data,
                               bool debug)
    : impl_(std::make_unique<Impl>(lexicon, tokens, dict_dir, meta_data,
                                   debug)) {}

MeloTtsLexicon::MeloTtsLexicon(const std::string &lexicon,
                               const std::string &tokens,
                               const OfflineTtsVitsModelMetaData &meta_data,
                               bool debug)
    : impl_(std::make_unique<Impl>(lexicon, tokens, meta_data, debug)) {}

template <typename Manager>
MeloTtsLexicon::MeloTtsLexicon(Manager *mgr, const std::string &lexicon,
                               const std::string &tokens,
                               const std::string &dict_dir,
                               const OfflineTtsVitsModelMetaData &meta_data,
                               bool debug)
    : impl_(std::make_unique<Impl>(mgr, lexicon, tokens, dict_dir, meta_data,
                                   debug)) {}

template <typename Manager>
MeloTtsLexicon::MeloTtsLexicon(Manager *mgr, const std::string &lexicon,
                               const std::string &tokens,
                               const OfflineTtsVitsModelMetaData &meta_data,
                               bool debug)
    : impl_(std::make_unique<Impl>(mgr, lexicon, tokens, meta_data, debug)) {}

std::vector<TokenIDs> MeloTtsLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string & /*unused_voice = ""*/) const {
  return impl_->ConvertTextToTokenIds(text);
}

#if __ANDROID_API__ >= 9
template MeloTtsLexicon::MeloTtsLexicon(
    AAssetManager *mgr, const std::string &lexicon, const std::string &tokens,
    const std::string &dict_dir, const OfflineTtsVitsModelMetaData &meta_data,
    bool debug);

template MeloTtsLexicon::MeloTtsLexicon(
    AAssetManager *mgr, const std::string &lexicon, const std::string &tokens,
    const OfflineTtsVitsModelMetaData &meta_data, bool debug);
#endif

#if __OHOS__
template MeloTtsLexicon::MeloTtsLexicon(
    NativeResourceManager *mgr, const std::string &lexicon,
    const std::string &tokens, const std::string &dict_dir,
    const OfflineTtsVitsModelMetaData &meta_data, bool debug);

template MeloTtsLexicon::MeloTtsLexicon(
    NativeResourceManager *mgr, const std::string &lexicon,
    const std::string &tokens, const OfflineTtsVitsModelMetaData &meta_data,
    bool debug);
#endif

}  // namespace sherpa_onnx
