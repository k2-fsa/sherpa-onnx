// sherpa-onnx/csrc/keyword-spotter-impl.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/keyword-spotter-impl.h"

#include "sherpa-onnx/csrc/keyword-spotter-transducer-impl.h"

#if SHERPA_ONNX_ENABLE_RKNN
#include "sherpa-onnx/csrc/keyword-spotter-transducer-rknn-impl.h"
#endif

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

namespace sherpa_onnx {

std::unique_ptr<KeywordSpotterImpl> KeywordSpotterImpl::Create(
    const KeywordSpotterConfig &config) {
  if (config.model_config.provider_config.provider == "rknn") {
#if SHERPA_ONNX_ENABLE_RKNN
    if (!config.model_config.transducer.encoder.empty()) {
      return std::make_unique<KeywordSpotterTransducerRknnImpl>(config);
    }
#else
    SHERPA_ONNX_LOGE(
        "Please rebuild sherpa-onnx with -DSHERPA_ONNX_ENABLE_RKNN=ON if you "
        "want to use rknn. Fallback to CPU. Make sure you pass an onnx model.");
#endif
  }

  if (!config.model_config.transducer.encoder.empty()) {
    return std::make_unique<KeywordSpotterTransducerImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  SHERPA_ONNX_EXIT(-1);
}

template <typename Manager>
std::unique_ptr<KeywordSpotterImpl> KeywordSpotterImpl::Create(
    Manager *mgr, const KeywordSpotterConfig &config) {
  if (config.model_config.provider_config.provider == "rknn") {
#if SHERPA_ONNX_ENABLE_RKNN
    if (!config.model_config.transducer.encoder.empty()) {
      return std::make_unique<KeywordSpotterTransducerRknnImpl>(mgr, config);
    }
#else
    SHERPA_ONNX_LOGE(
        "Please rebuild sherpa-onnx with -DSHERPA_ONNX_ENABLE_RKNN=ON if you "
        "want to use rknn. Fallback to CPU. Make sure you pass an onnx model.");
#endif
  }

  if (!config.model_config.transducer.encoder.empty()) {
    return std::make_unique<KeywordSpotterTransducerImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  exit(-1);
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<KeywordSpotterImpl> KeywordSpotterImpl::Create(
    AAssetManager *mgr, const KeywordSpotterConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<KeywordSpotterImpl> KeywordSpotterImpl::Create(
    NativeResourceManager *mgr, const KeywordSpotterConfig &config);
#endif

}  // namespace sherpa_onnx
