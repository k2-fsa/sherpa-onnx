// sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSpeechDenoiserGtcrnModelConfig::Register(ParseOptions *po) {
  po->Register("speech-denoiser-gtcrn-model", &model,
               "Path to the gtcrn model for speech denoising");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateGtcrnModelConfig(
    const OfflineSpeechDenoiserGtcrnModelConfig &config, const Exists &exists) {
  if (config.model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --speech-denoiser-gtcrn-model");
    return false;
  }

  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("gtcrn model file '%s' does not exist",
                     config.model.c_str());
    return false;
  }
  return true;
}

bool OfflineSpeechDenoiserGtcrnModelConfig::Validate() const {
  return ValidateGtcrnModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflineSpeechDenoiserGtcrnModelConfig::Validate(Manager *mgr) const {
  return ValidateGtcrnModelConfig(*this, [mgr](const std::string &p) -> bool {
    return FileExists(mgr, p);
  });
}

#if __ANDROID_API__ >= 9
template bool OfflineSpeechDenoiserGtcrnModelConfig::Validate(
    AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineSpeechDenoiserGtcrnModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineSpeechDenoiserGtcrnModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeechDenoiserGtcrnModelConfig(";
  os << "model=\"" << model << "\")";
  return os.str();
}

}  // namespace sherpa_onnx
