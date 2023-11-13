// sherpa-onnx/csrc/lexicon.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_LEXICON_H_
#define SHERPA_ONNX_CSRC_LEXICON_H_

#include <cstdint>
#include <memory>
#include <regex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

namespace sherpa_onnx {

// TODO(fangjun): Refactor it to an abstract class
class Lexicon {
 public:
  Lexicon(const std::string &lexicon, const std::string &tokens,
          const std::string &punctuations, const std::string &language,
          bool debug = false, bool is_piper = false);

#if __ANDROID_API__ >= 9
  Lexicon(AAssetManager *mgr, const std::string &lexicon,
          const std::string &tokens, const std::string &punctuations,
          const std::string &language, bool debug = false,
          bool is_piper = false);
#endif

  std::vector<int64_t> ConvertTextToTokenIds(const std::string &text) const;

 private:
  std::vector<int64_t> ConvertTextToTokenIdsGerman(
      const std::string &text) const {
    return ConvertTextToTokenIdsEnglish(text);
  }

  std::vector<int64_t> ConvertTextToTokenIdsSpanish(
      const std::string &text) const {
    return ConvertTextToTokenIdsEnglish(text);
  }

  std::vector<int64_t> ConvertTextToTokenIdsFrench(
      const std::string &text) const {
    return ConvertTextToTokenIdsEnglish(text);
  }

  std::vector<int64_t> ConvertTextToTokenIdsEnglish(
      const std::string &text) const;

  std::vector<int64_t> ConvertTextToTokenIdsChinese(
      const std::string &text) const;

  void InitLanguage(const std::string &lang);
  void InitTokens(std::istream &is);
  void InitLexicon(std::istream &is);
  void InitPunctuations(const std::string &punctuations);

 private:
  enum class Language {
    kEnglish,
    kGerman,
    kSpanish,
    kFrench,
    kChinese,
    kUnknown,
  };

 private:
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;
  std::unordered_set<std::string> punctuations_;
  std::unordered_map<std::string, int32_t> token2id_;
  Language language_;
  bool debug_;
  bool is_piper_;

  // for Chinese polyphones
  std::unique_ptr<std::regex> pattern_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_LEXICON_H_
