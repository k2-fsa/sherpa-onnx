// sherpa-onnx/csrc/offline-recognizer-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {

class OfflineRecognizerImpl {
 public:
  static std::unique_ptr<OfflineRecognizerImpl> Create(
      const OfflineRecognizerConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<OfflineRecognizerImpl> Create(
      AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

  virtual ~OfflineRecognizerImpl() = default;

  virtual std::unique_ptr<OfflineStream> CreateStream(
      const std::string &hotwords) const {
    SHERPA_ONNX_LOGE("Only transducer models support contextual biasing.");
    exit(-1);
  }

  virtual std::unique_ptr<OfflineStream> CreateStream() const = 0;

  virtual void DecodeStreams(OfflineStream **ss, int32_t n) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_IMPL_H_
