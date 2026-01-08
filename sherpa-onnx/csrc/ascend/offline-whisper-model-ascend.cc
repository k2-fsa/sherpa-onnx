// sherpa-onnx/csrc/ascend/offline-whisper-model-ascend.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include "sherpa-onnx/csrc/ascend/offline-whisper-model-ascend.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/ascend/macros.h"
#include "sherpa-onnx/csrc/ascend/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineWhisperModelAscend::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {}
  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {}

  std::vector<int32_t> Run(std::vector<float> features) { return {}; }

  int32_t FeatureDim() const { return 1; }

 private:
  std::mutex mutex_;
  Acl acl_;

  std::unique_ptr<AclContext> context_;

  OfflineModelConfig config_;
};

OfflineWhisperModelAscend::OfflineWhisperModelAscend(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineWhisperModelAscend::OfflineWhisperModelAscend(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineWhisperModelAscend::~OfflineWhisperModelAscend() = default;

std::vector<int32_t> OfflineWhisperModelAscend::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

int32_t OfflineWhisperModelAscend::FeatureDim() const {
  return impl_->FeatureDim();
}

#if __ANDROID_API__ >= 9
template OfflineWhisperModelAscend::OfflineWhisperModelAscend(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineWhisperModelAscend::OfflineWhisperModelAscend(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
