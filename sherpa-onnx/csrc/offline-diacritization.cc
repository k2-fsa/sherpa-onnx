// sherpa-onnx/csrc/offline-diacritization.cc
//
// Copyright (c)  2026  Matias Lin

#include "sherpa-onnx/csrc/offline-diacritization.h"

#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/offline-diacritization-impl.h"

namespace sherpa_onnx {

void OfflineDiacritizationConfig::Register(ParseOptions *po) {
  model.Register(po);
}

bool OfflineDiacritizationConfig::Validate() const {
  if (!model.Validate()) {
    return false;
  }
  return true;
}

template <typename Manager>
bool OfflineDiacritizationConfig::Validate(Manager *mgr) const {
  if (!model.Validate(mgr)) {
    return false;
  }
  return true;
}

#if __ANDROID_API__ >= 9
template bool OfflineDiacritizationConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineDiacritizationConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineDiacritizationConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineDiacritizationConfig(";
  os << "model=" << model.ToString() << ")";
  return os.str();
}

OfflineDiacritization::OfflineDiacritization(
    const OfflineDiacritizationConfig &config)
    : impl_(OfflineDiacritizationImpl::Create(config)) {}

template <typename Manager>
OfflineDiacritization::OfflineDiacritization(
    Manager *mgr, const OfflineDiacritizationConfig &config)
    : impl_(OfflineDiacritizationImpl::Create(mgr, config)) {}

#if __ANDROID_API__ >= 9
template OfflineDiacritization::OfflineDiacritization(
    AAssetManager *mgr, const OfflineDiacritizationConfig &config);
#endif

#if __OHOS__
template OfflineDiacritization::OfflineDiacritization(
    NativeResourceManager *mgr, const OfflineDiacritizationConfig &config);
#endif

OfflineDiacritization::~OfflineDiacritization() = default;

std::string OfflineDiacritization::AddDiacritics(
    const std::string &text) const {
  if (!impl_) {
    return text;
  }
  return impl_->AddDiacritics(text);
}

}  // namespace sherpa_onnx
