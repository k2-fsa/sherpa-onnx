// sherpa-onnx/csrc/offline-source-separation.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-source-separation.h"

#include <memory>

#include "sherpa-onnx/csrc/offline-source-separation-impl.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

namespace sherpa_onnx {

void OfflineSourceSeparationConfig::Register(ParseOptions *po) {
  model.Register(po);
}

bool OfflineSourceSeparationConfig::Validate() const {
  return model.Validate();
}

std::string OfflineSourceSeparationConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSourceSeparationConfig(";
  os << "model=" << model.ToString() << ")";

  return os.str();
}

template <typename Manager>
OfflineSourceSeparation::OfflineSourceSeparation(
    Manager *mgr, const OfflineSourceSeparationConfig &config)
    : impl_(OfflineSourceSeparationImpl::Create(mgr, config)) {}

OfflineSourceSeparation::OfflineSourceSeparation(
    const OfflineSourceSeparationConfig &config)
    : impl_(OfflineSourceSeparationImpl::Create(config)) {}

OfflineSourceSeparation::~OfflineSourceSeparation() = default;

OfflineSourceSeparationOutput OfflineSourceSeparation::Process(
    const OfflineSourceSeparationInput &input) const {
  return impl_->Process(input);
}

int32_t OfflineSourceSeparation::GetOutputSampleRate() const {
  return impl_->GetOutputSampleRate();
}

// e.g., it is 2 for 2stems from spleeter
int32_t OfflineSourceSeparation::GetNumberOfStems() const {
  return impl_->GetNumberOfStems();
}

#if __ANDROID_API__ >= 9
template OfflineSourceSeparation::OfflineSourceSeparation(
    AAssetManager *mgr, const OfflineSourceSeparationConfig &config);
#endif

#if __OHOS__
template OfflineSourceSeparation::OfflineSourceSeparation(
    NativeResourceManager *mgr, const OfflineSourceSeparationConfig &config);
#endif

}  // namespace sherpa_onnx
