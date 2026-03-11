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

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-impl.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OfflineSpeechDenoiserImpl> OfflineSpeechDenoiserImpl::Create(
    const OfflineSpeechDenoiserConfig &config) {
  const bool has_gtcrn = !config.model.gtcrn.model.empty();
  const bool has_dpdfnet = !config.model.dpdfnet.model.empty();

  if (has_gtcrn && !has_dpdfnet) {
    return std::make_unique<OfflineSpeechDenoiserGtcrnImpl>(config);
  }

  if (has_dpdfnet && !has_gtcrn) {
    return std::make_unique<OfflineSpeechDenoiserDpdfNetImpl>(config);
  }

  if (has_gtcrn && has_dpdfnet) {
    SHERPA_ONNX_LOGE("Please provide only one speech denoising model.");
    return nullptr;
  }

  SHERPA_ONNX_LOGE("Please provide a speech denoising model.");
  return nullptr;
}

template <typename Manager>
std::unique_ptr<OfflineSpeechDenoiserImpl> OfflineSpeechDenoiserImpl::Create(
    Manager *mgr, const OfflineSpeechDenoiserConfig &config) {
  const bool has_gtcrn = !config.model.gtcrn.model.empty();
  const bool has_dpdfnet = !config.model.dpdfnet.model.empty();

  if (has_gtcrn && !has_dpdfnet) {
    return std::make_unique<OfflineSpeechDenoiserGtcrnImpl>(mgr, config);
  }

  if (has_dpdfnet && !has_gtcrn) {
    return std::make_unique<OfflineSpeechDenoiserDpdfNetImpl>(mgr, config);
  }

  if (has_gtcrn && has_dpdfnet) {
    SHERPA_ONNX_LOGE("Please provide only one speech denoising model.");
    return nullptr;
  }

  SHERPA_ONNX_LOGE("Please provide a speech denoising model.");
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
