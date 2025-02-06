// sherpa-onnx/csrc/kokoro-multi-lang-lexicon.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/kokoro-multi-lang-lexicon.h"

#include <codecvt>
#include <fstream>
#include <locale>
#include <regex>
#include <sstream>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "cppjieba/Jieba.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

static std::wstring ToWideString(const std::string &s) {
  // see
  // https://stackoverflow.com/questions/2573834/c-convert-string-or-char-to-wstring-or-wchar-t
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.from_bytes(s);
}

static std::string ToString(const std::wstring &s) {
  // see
  // https://stackoverflow.com/questions/2573834/c-convert-string-or-char-to-wstring-or-wchar-t
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.to_bytes(s);
}

class KokoroMultiLangLexicon::Impl {
 public:
  Impl(const std::string &tokens, const std::string &lexicon,
       const std::string &dict_dir, const std::string &data_dir,
       const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
      : meta_data_(meta_data), debug_(debug) {
    InitTokens(tokens);

    InitLexicon(lexicon);
    InitJieba(dict_dir);

    InitEspeak(data_dir);  // See ./piper-phonemize-lexicon.cc
    SHERPA_ONNX_LOGE("initialized!");
  }

  std::vector<TokenIDs> ConvertTextToTokenIds(const std::string &_text) const {
    std::string text = ToLowerCase(_text);
    if (debug_) {
      SHERPA_ONNX_LOGE("After converting to lowercase:\n%s", text.c_str());
    }

    std::vector<std::pair<std::string, std::string>> replace_str_pairs = {
        {"，", ","}, {":", ","},  {"、", ","}, {"；", ";"},   {"：", ":"},
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

    // https://en.cppreference.com/w/cpp/regex
    // https://stackoverflow.com/questions/37989081/how-to-use-unicode-range-in-c-regex
    std::string expr =
        "([;:,.?!'\"…\\(\\)“”])|([\\u4e00-\\u9fff]+)|([\\u0000-\\u007f]+)";

    auto ws = ToWideString(text);
    std::wstring wexpr = ToWideString(expr);
    std::wregex we(wexpr);

    auto begin = std::wsregex_iterator(ws.begin(), ws.end(), we);
    auto end = std::wsregex_iterator();

    std::vector<TokenIDs> ans;

    for (std::wsregex_iterator i = begin; i != end; ++i) {
      std::wsmatch match = *i;
      std::wstring match_str = match.str();
      auto ms = ToString(match_str);
      uint8_t c = reinterpret_cast<const uint8_t *>(ms.data())[0];
      if (c < 0x80) {
        if (debug_) {
          SHERPA_ONNX_LOGE("Non-Chinese: %s", ms.c_str());
        }
        auto ids = ConvertEnglishToTokenIDs(ms);
        ans.emplace_back(ids);
      } else {
        if (debug_) {
          SHERPA_ONNX_LOGE("Chinese: %s", ms.c_str());
        }
      }
    }

    return ans;
  }

 private:
  bool IsPunctuation(const std::string &text) const {
    if (text == ";" || text == ":" || text == "," || text == "." ||
        text == "!" || text == "?" || text == "—" || text == "…" ||
        text == "\"" || text == "(" || text == ")" || text == "“" ||
        text == "”") {
      return true;
    }

    return false;
  }

  std::vector<int32_t> ConvertChineseToTokenIDs(const std::string &text) const {
    return {};
  }
  std::vector<int32_t> ConvertEnglishToTokenIDs(const std::string &text) const {
    std::vector<int32_t> ans;

    if (IsPunctuation(text)) {
      ans.push_back(token2id_.at(text));
      return ans;
    }

    std::vector<std::string> words = SplitUtf8(text);
    if (debug_) {
      std::ostringstream os;
      os << "After splitting to words: ";
      std::string sep;
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }

    for (const auto &word : words) {
      if (IsPunctuation(word)) {
        SHERPA_ONNX_LOGE("Found punctuation: '%s'", word.c_str());
        ans.push_back(token2id_.at(word));
      } else if (word2ids_.count(word)) {
        const auto &ids = word2ids_.at(word);
        ans.insert(ans.end(), ids.begin(), ids.end());
      } else {
        SHERPA_ONNX_LOGE("Skip OOV: '%s'", word.c_str());
      }
    }

    return ans;
  }
  void InitTokens(const std::string &tokens) {
    std::ifstream is(tokens);
    InitTokens(is);
  }

  void InitTokens(std::istream &is) {
    token2id_ = ReadTokens(is);  // defined in ./symbol-table.cc
  }

  void InitLexicon(const std::string &lexicon) {
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
    std::string token;

    std::string line;
    int32_t line_num = 0;
    int32_t num_warn = 0;
    while (std::getline(is, line)) {
      ++line_num;
      std::istringstream iss(line);

      token_list.clear();
      iss >> word;
      ToLowerCase(&word);

      if (word2ids_.count(word)) {
        num_warn += 1;
        if (num_warn < 10) {
          SHERPA_ONNX_LOGE("Duplicated word: %s at line %d:%s. Ignore it.",
                           word.c_str(), line_num, line.c_str());
        }
        continue;
      }

      while (iss >> token) {
        token_list.push_back(std::move(token));
      }

      std::vector<int32_t> ids = ConvertTokensToIds(token2id_, token_list);

      if (ids.empty()) {
        SHERPA_ONNX_LOGE(
            "Invalid pronunciation for word '%s' at line %d:%s. Ignore it",
            word.c_str(), line_num, line.c_str());
        continue;
      }

      word2ids_.insert({std::move(word), std::move(ids)});
    }
  }

  void InitJieba(const std::string &dict_dir) {
    std::string dict = dict_dir + "/jieba.dict.utf8";
    std::string hmm = dict_dir + "/hmm_model.utf8";
    std::string user_dict = dict_dir + "/user.dict.utf8";
    std::string idf = dict_dir + "/idf.utf8";
    std::string stop_word = dict_dir + "/stop_words.utf8";

    AssertFileExists(dict);
    AssertFileExists(hmm);
    AssertFileExists(user_dict);
    AssertFileExists(idf);
    AssertFileExists(stop_word);

    jieba_ =
        std::make_unique<cppjieba::Jieba>(dict, hmm, user_dict, idf, stop_word);
  }

 private:
  OfflineTtsKokoroModelMetaData meta_data_;

  // word to token IDs
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;

  // tokens.txt is saved in token2id_
  std::unordered_map<std::string, int32_t> token2id_;

  std::unique_ptr<cppjieba::Jieba> jieba_;
  bool debug_ = false;
};

KokoroMultiLangLexicon::~KokoroMultiLangLexicon() = default;

KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    const std::string &tokens, const std::string &lexicon,
    const std::string &dict_dir, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
    : impl_(std::make_unique<Impl>(tokens, lexicon, dict_dir, data_dir,
                                   meta_data, debug)) {}

template <typename Manager>
KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    Manager *mgr, const std::string &tokens, const std::string &lexicon,
    const std::string &dict_dir, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
    : impl_(std::make_unique<Impl>(mgr, tokens, lexicon, dict_dir, data_dir,
                                   meta_data, debug)) {}

std::vector<TokenIDs> KokoroMultiLangLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string & /*unused_voice = ""*/) const {
  return impl_->ConvertTextToTokenIds(text);
}

#if __ANDROID_API__ >= 9
template KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &lexicon,
    const std::string &dict_dir, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &meta_data, bool debug);
#endif

#if __OHOS__
template KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &lexicon, const std::string &dict_dir,
    const std::string &data_dir, const OfflineTtsKokoroModelMetaData &meta_data,
    bool debug);
#endif

}  // namespace sherpa_onnx
