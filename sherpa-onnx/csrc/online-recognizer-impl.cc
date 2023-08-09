// sherpa-onnx/csrc/online-recognizer-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-recognizer-impl.h"

#include "sherpa-onnx/csrc/online-recognizer-transducer-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OnlineRecognizerImpl> OnlineRecognizerImpl::Create(
    const OnlineRecognizerConfig &config) {
  if (!config.model_config.transducer.encoder.empty()) {
    return std::make_unique<OnlineRecognizerTransducerImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  exit(-1);
}

#if __ANDROID_API__ >= 9
std::unique_ptr<OnlineRecognizerImpl> OnlineRecognizerImpl::Create(
    AAssetManager *mgr, const OnlineRecognizerConfig &config) {
  if (!config.model_config.transducer.encoder.empty()) {
    return std::make_unique<OnlineRecognizerTransducerImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  exit(-1);
}
#endif

}  // namespace sherpa_onnx
