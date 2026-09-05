// sherpa-onnx/csrc/online-speech-denoiser-impl.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-speech-denoiser-impl.h"

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-speech-denoiser-dpdfnet-impl.h"
#include "sherpa-onnx/csrc/online-speech-denoiser-gtcrn-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OnlineSpeechDenoiserImpl> OnlineSpeechDenoiserImpl::Create(
    const OnlineSpeechDenoiserConfig &config) {
  const bool has_gtcrn = !config.model.gtcrn.model.empty();
  const bool has_dpdfnet = !config.model.dpdfnet.model.empty();

  if (has_gtcrn) {
    return std::make_unique<OnlineSpeechDenoiserGtcrnImpl>(config);
  } else if (has_dpdfnet) {
    return std::make_unique<OnlineSpeechDenoiserDpdfNetImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please provide one speech denoising model.");
  return nullptr;
}

template <typename Manager>
std::unique_ptr<OnlineSpeechDenoiserImpl> OnlineSpeechDenoiserImpl::Create(
    Manager *mgr, const OnlineSpeechDenoiserConfig &config) {
  const bool has_gtcrn = !config.model.gtcrn.model.empty();
  const bool has_dpdfnet = !config.model.dpdfnet.model.empty();

  if (has_gtcrn) {
    return std::make_unique<OnlineSpeechDenoiserGtcrnImpl>(mgr, config);
  } else if (has_dpdfnet) {
    return std::make_unique<OnlineSpeechDenoiserDpdfNetImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please provide one speech denoising model.");
  return nullptr;
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OnlineSpeechDenoiserImpl>
OnlineSpeechDenoiserImpl::Create(AAssetManager *mgr,
                                 const OnlineSpeechDenoiserConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OnlineSpeechDenoiserImpl>
OnlineSpeechDenoiserImpl::Create(NativeResourceManager *mgr,
                                 const OnlineSpeechDenoiserConfig &config);
#endif

}  // namespace sherpa_onnx
