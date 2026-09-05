// sherpa-onnx/csrc/offline-diacritization-impl.cc
//
// Copyright (c)  2026  Matias Lin

#include "sherpa-onnx/csrc/offline-diacritization-impl.h"

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-diacritization-catt-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OfflineDiacritizationImpl> OfflineDiacritizationImpl::Create(
    const OfflineDiacritizationConfig &config) {
  if (config.model.catt_encoder.empty()) {
    SHERPA_ONNX_LOGE(
        "Please specify a diacritization encoder model. Return a null pointer");
    return nullptr;
  }
  if (config.model.catt_decoder.empty()) {
    SHERPA_ONNX_LOGE(
        "Please specify a diacritization decoder model. Return a null pointer");
    return nullptr;
  }
  return std::make_unique<OfflineDiacritizationCATTImpl>(config);
}

template <typename Manager>
std::unique_ptr<OfflineDiacritizationImpl> OfflineDiacritizationImpl::Create(
    Manager *mgr, const OfflineDiacritizationConfig &config) {
  if (config.model.catt_encoder.empty()) {
    SHERPA_ONNX_LOGE(
        "Please specify a diacritization encoder model. Return a null pointer");
    return nullptr;
  }
  if (config.model.catt_decoder.empty()) {
    SHERPA_ONNX_LOGE(
        "Please specify a diacritization decoder model. Return a null pointer");
    return nullptr;
  }
  return std::make_unique<OfflineDiacritizationCATTImpl>(mgr, config);
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineDiacritizationImpl>
OfflineDiacritizationImpl::Create(AAssetManager *mgr,
                                  const OfflineDiacritizationConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflineDiacritizationImpl>
OfflineDiacritizationImpl::Create(NativeResourceManager *mgr,
                                  const OfflineDiacritizationConfig &config);
#endif

}  // namespace sherpa_onnx
