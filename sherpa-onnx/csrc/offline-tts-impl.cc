// sherpa-onnx/csrc/offline-tts-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-impl.h"

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/offline-tts-matcha-impl.h"
#include "sherpa-onnx/csrc/offline-tts-vits-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    const OfflineTtsConfig &config) {
  if (!config.model.vits.model.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(config);
  }
  return std::make_unique<OfflineTtsMatchaImpl>(config);
}

template <typename Manager>
std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    Manager *mgr, const OfflineTtsConfig &config) {
  if (!config.model.vits.model.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(mgr, config);
  }

  return std::make_unique<OfflineTtsMatchaImpl>(mgr, config);
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    AAssetManager *mgr, const OfflineTtsConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    NativeResourceManager *mgr, const OfflineTtsConfig &config);
#endif

}  // namespace sherpa_onnx
