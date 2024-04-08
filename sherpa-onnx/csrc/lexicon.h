// sherpa-onnx/csrc/lexicon.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_LEXICON_H_
#define SHERPA_ONNX_CSRC_LEXICON_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/offline-tts-frontend.h"

namespace sherpa_onnx {

class Lexicon : public OfflineTtsFrontend {
 public:
  Lexicon() = default;  // for subclasses
                        //
  // Note: for models from piper, we won't use this class.
  Lexicon(const std::string &lexicon, const std::string &tokens,
          const std::string &punctuations, const std::string &language,
          bool debug = false);

#if __ANDROID_API__ >= 9
  Lexicon(AAssetManager *mgr, const std::string &lexicon,
          const std::string &tokens, const std::string &punctuations,
          const std::string &language, bool debug = false);
#endif

  std::vector<std::vector<int64_t>> ConvertTextToTokenIds(
      const std::string &text, const std::string &voice = "") const override;

 private:
  std::vector<std::vector<int64_t>> ConvertTextToTokenIdsNotChinese(
      const std::string &text) const;

  std::vector<std::vector<int64_t>> ConvertTextToTokenIdsChinese(
      const std::string &text) const;

  void InitLanguage(const std::string &lang);
  void InitTokens(std::istream &is);
  void InitLexicon(std::istream &is);
  void InitPunctuations(const std::string &punctuations);

 private:
  enum class Language {
    kNotChinese,
    kChinese,
    kUnknown,
  };

 private:
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;
  std::unordered_set<std::string> punctuations_;
  std::unordered_map<std::string, int32_t> token2id_;
  Language language_;
  bool debug_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_LEXICON_H_
