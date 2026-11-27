// sherpa-onnx/csrc/audio-tagging.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/audio-tagging.h"

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/audio-tagging-impl.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

std::string AudioEvent::ToString() const {
  std::ostringstream os;
  os << "AudioEvent(";
  os << "name=\"" << name << "\", ";
  os << "index=" << index << ", ";
  os << "prob=" << prob << ")";
  return os.str();
}

void AudioTaggingConfig::Register(ParseOptions *po) {
  model.Register(po);
  po->Register("labels", &labels, "Event label file");
  po->Register("top-k", &top_k, "Top k events to return in the result");
}

// 'validate_model' and 'exists' check the model config and resolve a path
// either against the file system or against a resource manager.
template <typename ValidateModel, typename Exists>
static bool ValidateAudioTaggingConfig(const AudioTaggingConfig &config,
                                       const ValidateModel &validate_model,
                                       const Exists &exists) {
  if (!validate_model()) {
    return false;
  }

  if (config.top_k < 1) {
    SHERPA_ONNX_LOGE("--top-k should be >= 1. Given: %d", config.top_k);
    return false;
  }

  if (config.labels.empty()) {
    SHERPA_ONNX_LOGE("Please provide --labels");
    return false;
  }

  if (!exists(config.labels)) {
    SHERPA_ONNX_LOGE("--labels '%s' does not exist", config.labels.c_str());
    return false;
  }

  return true;
}

bool AudioTaggingConfig::Validate() const {
  return ValidateAudioTaggingConfig(
      *this, [this]() -> bool { return model.Validate(); },
      [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool AudioTaggingConfig::Validate(Manager *mgr) const {
  return ValidateAudioTaggingConfig(
      *this, [this, mgr]() -> bool { return model.Validate(mgr); },
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool AudioTaggingConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool AudioTaggingConfig::Validate(NativeResourceManager *mgr) const;
#endif
std::string AudioTaggingConfig::ToString() const {
  std::ostringstream os;

  os << "AudioTaggingConfig(";
  os << "model=" << model.ToString() << ", ";
  os << "labels=\"" << labels << "\", ";
  os << "top_k=" << top_k << ")";

  return os.str();
}

AudioTagging::AudioTagging(const AudioTaggingConfig &config)
    : impl_(AudioTaggingImpl::Create(config)) {}

#if __ANDROID_API__ >= 9
AudioTagging::AudioTagging(AAssetManager *mgr, const AudioTaggingConfig &config)
    : impl_(AudioTaggingImpl::Create(mgr, config)) {}
#endif

AudioTagging::~AudioTagging() = default;

std::unique_ptr<OfflineStream> AudioTagging::CreateStream() const {
  return impl_->CreateStream();
}

std::vector<AudioEvent> AudioTagging::Compute(OfflineStream *s,
                                              int32_t top_k /*= -1*/) const {
  return impl_->Compute(s, top_k);
}

}  // namespace sherpa_onnx
