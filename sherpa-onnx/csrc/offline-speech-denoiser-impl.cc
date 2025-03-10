// sherpa-onnx/csrc/offline-speech-denoiser-impl.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-onnx/csrc/offline-speech-denoiser-impl.h"

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

namespace sherpa_onnx {

std::unique_ptr<OfflineSpeechDenoiserImpl> OfflineSpeechDenoiserImpl::Create(
    const OfflineSpeechDenoiserConfig &config) {
  return nullptr;
}

template <typename Manager>
std::unique_ptr<OfflineSpeechDenoiserImpl> OfflineSpeechDenoiserImpl::Create(
    Manager *mgr, const OfflineSpeechDenoiserConfig &config) {
  return nullptr;
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineSpeechDenoiserImpl>
OfflineSpeechDenoiserImpl::Create(AAssetManager *mgr,
                                  const OfflineSpeechDenoiserConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflineSpeechDenoiserImpl>
OfflineSpeechDenoiserImpl::Create(NativeResourceManager *mgr,
                                  const OfflineSpeechDenoiserConfig &config);
#endif

}  // namespace sherpa_onnx
