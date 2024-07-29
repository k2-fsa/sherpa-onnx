// sherpa-onnx/csrc/melo-tts-lexicon.cc
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/melo-tts-lexicon.h"

#include <fstream>
#include <regex>  // NOLINT
#include <utility>

#include "cppjieba/Jieba.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

// implemented in ./lexicon.cc
std::unordered_map<std::string, int32_t> ReadTokens(std::istream &is);

std::vector<int32_t> ConvertTokensToIds(
    const std::unordered_map<std::string, int32_t> &token2id,
    const std::vector<std::string> &tokens);

class MeloTtsLexicon::Impl {
 public:
  Impl(const std::string &lexicon, const std::string &tokens,
       const std::string &dict_dir,
       const OfflineTtsVitsModelMetaData &meta_data, bool debug)
      : meta_data_(meta_data), debug_(debug) {
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

    {
      std::ifstream is(tokens);
      InitTokens(is);
    }

    {
      std::ifstream is(lexicon);
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

    std::vector<std::string> words;
    bool is_hmm = true;
    jieba_->Cut(text, words, is_hmm);

    if (debug_) {
      SHERPA_ONNX_LOGE("input text: %s", text.c_str());
      SHERPA_ONNX_LOGE("after replacing punctuations: %s", s.c_str());

      std::ostringstream os;
      std::string sep = "";
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }

      SHERPA_ONNX_LOGE("after jieba processing: %s", os.str().c_str());
    }

    std::vector<TokenIDs> ans;
    TokenIDs this_sentence;

    for (const auto &w : words) {
      auto ids = ConvertWordToIds(w);
      if (ids.tokens.empty()) {
        SHERPA_ONNX_LOGE("Ignore OOV '%s'", w.c_str());
        continue;
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
    }  // for (const auto &w : words)

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
      }
    }

    return ans;
  }

  void InitTokens(std::istream &is) {
    token2id_ = ReadTokens(is);
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

    word2ids_["呣"] = word2ids_["母"];
    word2ids_["嗯"] = word2ids_["恩"];
  }

 private:
  // lexicon.txt is saved in word2ids_
  std::unordered_map<std::string, TokenIDs> word2ids_;

  // tokens.txt is saved in token2id_
  std::unordered_map<std::string, int32_t> token2id_;

  OfflineTtsVitsModelMetaData meta_data_;

  std::unique_ptr<cppjieba::Jieba> jieba_;
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

std::vector<TokenIDs> MeloTtsLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string & /*unused_voice = ""*/) const {
  return impl_->ConvertTextToTokenIds(text);
}

}  // namespace sherpa_onnx
