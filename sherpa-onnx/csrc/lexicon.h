// sherpa-onnx/csrc/lexicon.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_LEXICON_H_
#define SHERPA_ONNX_CSRC_LEXICON_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sherpa_onnx {

// TODO(fangjun): Refactor it to an abstract class
class Lexicon {
 public:
  Lexicon(const std::string &lexicon, const std::string &tokens,
          const std::string &punctuations, const std::string &language);

  std::vector<int64_t> ConvertTextToTokenIds(const std::string &text) const;

 private:
  std::vector<int64_t> ConvertTextToTokenIdsEnglish(
      const std::string &text) const;

  std::vector<int64_t> ConvertTextToTokenIdsChinese(
      const std::string &text) const;

  void InitLanguage(const std::string &lang);
  void InitTokens(const std::string &tokens);
  void InitLexicon(const std::string &lexicon);
  void InitPunctuations(const std::string &punctuations);

 private:
  enum class Language {
    kEnglish,
    kChinese,
    kUnknown,
  };

 private:
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;
  std::unordered_set<std::string> punctuations_;
  std::unordered_map<std::string, int32_t> token2id_;
  Language language_;
  //
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_LEXICON_H_
