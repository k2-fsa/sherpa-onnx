// sherpa-onnx/csrc/offline-tts-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-impl.h"

#include <memory>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/offline-tts-kitten-impl.h"
#include "sherpa-onnx/csrc/offline-tts-kokoro-impl.h"
#include "sherpa-onnx/csrc/offline-tts-matcha-impl.h"
#include "sherpa-onnx/csrc/offline-tts-vits-impl.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-impl.h"

namespace sherpa_onnx {

std::vector<int64_t> OfflineTtsImpl::AddBlank(const std::vector<int64_t> &x,
                                              int32_t blank_id /*= 0*/) const {
  // we assume the blank ID is 0
  std::vector<int64_t> buffer(x.size() * 2 + 1, blank_id);
  int32_t i = 1;
  for (auto k : x) {
    buffer[i] = k;
    i += 2;
  }
  return buffer;
}

std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    const OfflineTtsConfig &config) {
  if (!config.model.vits.model.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(config);
  } else if (!config.model.matcha.acoustic_model.empty()) {
    return std::make_unique<OfflineTtsMatchaImpl>(config);
  } else if (!config.model.zipvoice.text_model.empty() &&
             !config.model.zipvoice.flow_matching_model.empty()) {
    return std::make_unique<OfflineTtsZipvoiceImpl>(config);
  } else if (!config.model.kokoro.model.empty()) {
    return std::make_unique<OfflineTtsKokoroImpl>(config);
  } else if (!config.model.kitten.model.empty()) {
    return std::make_unique<OfflineTtsKittenImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please provide a tts model.");

  return {};
}

template <typename Manager>
std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    Manager *mgr, const OfflineTtsConfig &config) {
  if (!config.model.vits.model.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(mgr, config);
  } else if (!config.model.matcha.acoustic_model.empty()) {
    return std::make_unique<OfflineTtsMatchaImpl>(mgr, config);
  } else if (!config.model.zipvoice.text_model.empty() &&
             !config.model.zipvoice.flow_matching_model.empty()) {
    return std::make_unique<OfflineTtsZipvoiceImpl>(mgr, config);
  } else if (!config.model.kokoro.model.empty()) {
    return std::make_unique<OfflineTtsKokoroImpl>(mgr, config);
  } else if (!config.model.kitten.model.empty()) {
    return std::make_unique<OfflineTtsKittenImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please provide a tts model.");
  return {};
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
