// sherpa-onnx/csrc/jieba-lexicon.h
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_JIEBA_LEXICON_H_
#define SHERPA_ONNX_CSRC_JIEBA_LEXICON_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model-metadata.h"

namespace sherpa_onnx {

class JiebaLexicon : public OfflineTtsFrontend {
 public:
  ~JiebaLexicon() override;
  JiebaLexicon(const std::string &lexicon, const std::string &tokens,
               const std::string &dict_dir,
               const OfflineTtsVitsModelMetaData &meta_data, bool debug);

#if __ANDROID_API__ >= 9
  JiebaLexicon(AAssetManager *mgr, const std::string &lexicon,
               const std::string &tokens, const std::string &dict_dir,
               const OfflineTtsVitsModelMetaData &meta_data);
#endif

  std::vector<std::vector<int64_t>> ConvertTextToTokenIds(
      const std::string &text,
      const std::string &unused_voice = "") const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_JIEBA_LEXICON_H_
