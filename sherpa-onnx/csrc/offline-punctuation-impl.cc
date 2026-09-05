// sherpa-onnx/csrc/offline-punctuation-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-punctuation-impl.h"

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-punctuation-ct-transformer-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OfflinePunctuationImpl> OfflinePunctuationImpl::Create(
    const OfflinePunctuationConfig &config) {
  if (!config.model.ct_transformer.empty()) {
    return std::make_unique<OfflinePunctuationCtTransformerImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please specify a punctuation model! Return a null pointer");
  return nullptr;
}

template <typename Manager>
std::unique_ptr<OfflinePunctuationImpl> OfflinePunctuationImpl::Create(
    Manager *mgr, const OfflinePunctuationConfig &config) {
  if (!config.model.ct_transformer.empty()) {
    return std::make_unique<OfflinePunctuationCtTransformerImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please specify a punctuation model! Return a null pointer");
  return nullptr;
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflinePunctuationImpl> OfflinePunctuationImpl::Create(
    AAssetManager *mgr, const OfflinePunctuationConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflinePunctuationImpl> OfflinePunctuationImpl::Create(
    NativeResourceManager *mgr, const OfflinePunctuationConfig &config);
#endif

}  // namespace sherpa_onnx
