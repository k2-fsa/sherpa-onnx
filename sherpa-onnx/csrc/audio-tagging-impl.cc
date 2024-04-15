// sherpa-onnx/csrc/audio-tagging-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/audio-tagging-impl.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/audio-tagging-zipformer-impl.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

std::unique_ptr<AudioTaggingImpl> AudioTaggingImpl::Create(
    const AudioTaggingConfig &config) {
  if (!config.model.zipformer.model.empty()) {
    return std::make_unique<AudioTaggingZipformerImpl>(config);
  }

  SHERPA_ONNX_LOG(
      "Please specify an audio tagging model! Return a null pointer");
  return nullptr;
}

#if __ANDROID_API__ >= 9
std::unique_ptr<AudioTaggingImpl> AudioTaggingImpl::Create(
    AAssetManager *mgr, const AudioTaggingConfig &config) {
  if (!config.model.zipformer.model.empty()) {
    return std::make_unique<AudioTaggingZipformerImpl>(mgr, config);
  }

  SHERPA_ONNX_LOG(
      "Please specify an audio tagging model! Return a null pointer");
  return nullptr;
}
#endif

}  // namespace sherpa_onnx
