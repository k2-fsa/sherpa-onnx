// sherpa-onnx/csrc/offline-tts-character-frontend.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_CHARACTER_FRONTEND_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_CHARACTER_FRONTEND_H_
#include <cstdint>
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

class OfflineTtsCharacterFrontend : public OfflineTtsFrontend {
 public:
  OfflineTtsCharacterFrontend(const std::string &tokens,
                              const OfflineTtsVitsModelMetaData &meta_data);

#if __ANDROID_API__ >= 9
  OfflineTtsCharacterFrontend(AAssetManager *mgr, const std::string &tokens,
                              const OfflineTtsVitsModelMetaData &meta_data);

#endif
  /** Convert a string to token IDs.
   *
   * @param text The input text.
   *             Example 1: "This is the first sample sentence; this is the
   *             second one." Example 2: "这是第一句。这是第二句。"
   * @param voice Optional. It is for espeak-ng.
   *
   * @return Return a vector-of-vector of token IDs. Each subvector contains
   *         a sentence that can be processed independently.
   *         If a frontend does not support splitting the text into
   * sentences, the resulting vector contains only one subvector.
   */
  std::vector<std::vector<int64_t>> ConvertTextToTokenIds(
      const std::string &text, const std::string &voice = "") const override;

 private:
  OfflineTtsVitsModelMetaData meta_data_;
  std::unordered_map<char32_t, int32_t> token2id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CHARACTER_FRONTEND_H_
