// sherpa-onnx/csrc/offline-source-separation-impl.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-source-separation-impl.h"

#include <memory>
#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/offline-source-separation-spleeter-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OfflineSourceSeparationImpl>
OfflineSourceSeparationImpl::Create(
    const OfflineSourceSeparationConfig &config) {
  // TODO(fangjun): Support other models
  return std::make_unique<OfflineSourceSeparationSpleeterImpl>(config);
}

template <typename Manager>
std::unique_ptr<OfflineSourceSeparationImpl>
OfflineSourceSeparationImpl::Create(
    Manager *mgr, const OfflineSourceSeparationConfig &config) {
  // TODO(fangjun): Support other models
  return std::make_unique<OfflineSourceSeparationSpleeterImpl>(mgr, config);
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineSourceSeparationImpl>
OfflineSourceSeparationImpl::Create(
    AAssetManager *mgr, const OfflineSourceSeparationConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflineSourceSeparationImpl>
OfflineSourceSeparationImpl::Create(
    NativeResourceManager *mgr, const OfflineSourceSeparationConfig &config);
#endif

}  // namespace sherpa_onnx
