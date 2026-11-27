// sherpa-onnx/csrc/offline-zipformer-audio-tagging-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-zipformer-audio-tagging-model-config.h"

#include <memory>
#include <string>
#include <utility>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineZipformerAudioTaggingModelConfig::Register(ParseOptions *po) {
  po->Register("zipformer-model", &model,
               "Path to zipformer model for audio tagging");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOfflineZipformerAudioTaggingModelConfig(
    const OfflineZipformerAudioTaggingModelConfig &config,
    const Exists &exists) {
  if (config.model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipformer-model");
    return false;
  }

  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("--zipformer-model: '%s' does not exist",
                     config.model.c_str());
    return false;
  }

  return true;
}

bool OfflineZipformerAudioTaggingModelConfig::Validate() const {
  return ValidateOfflineZipformerAudioTaggingModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflineZipformerAudioTaggingModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflineZipformerAudioTaggingModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OfflineZipformerAudioTaggingModelConfig::Validate(
    AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineZipformerAudioTaggingModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineZipformerAudioTaggingModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineZipformerAudioTaggingModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
