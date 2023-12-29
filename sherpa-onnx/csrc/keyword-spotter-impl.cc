// sherpa-onnx/csrc/keyword-spotter-impl.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/keyword-spotter-impl.h"

#include "sherpa-onnx/csrc/keyword-spotter-transducer-impl.h"

namespace sherpa_onnx {

std::unique_ptr<KeywordSpotterImpl> KeywordSpotterImpl::Create(
    const KeywordSpotterConfig &config) {
  if (!config.model_config.transducer.encoder.empty()) {
    return std::make_unique<KeywordSpotterTransducerImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  exit(-1);
}

#if __ANDROID_API__ >= 9
std::unique_ptr<KeywordSpotterImpl> KeywordSpotterImpl::Create(
    AAssetManager *mgr, const KeywordSpotterConfig &config) {
  if (!config.model_config.transducer.encoder.empty()) {
    return std::make_unique<KeywordSpotterTransducerImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  exit(-1);
}
#endif

}  // namespace sherpa_onnx
