// sherpa-onnx/csrc/offline-tts-matcha-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-matcha-model.h"

#include <algorithm>
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

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

class OfflineTtsMatchaModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config) {}

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config) {}
};

OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsMatchaModel::~OfflineTtsMatchaModel() = default;

#if __ANDROID_API__ >= 9
template OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
