// sherpa-onnx/csrc/online-recognizer-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-recognizer-impl.h"

#include "sherpa-onnx/csrc/online-recognizer-ctc-impl.h"
#include "sherpa-onnx/csrc/online-recognizer-paraformer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer-transducer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer-transducer-nemo-impl.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

std::unique_ptr<OnlineRecognizerImpl> OnlineRecognizerImpl::Create(
    const OnlineRecognizerConfig &config) {
  
  if (!config.model_config.transducer.encoder.empty()) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
    
    auto decoder_model = ReadFile(config.model_config.transducer.decoder);
    auto sess = std::make_unique<Ort::Session>(env, decoder_model.data(), decoder_model.size(), Ort::SessionOptions{});
    
    size_t node_count = sess->GetOutputCount();
    
    if (node_count == 1) {
      return std::make_unique<OnlineRecognizerTransducerImpl>(config);
    } else {
      SHERPA_ONNX_LOGE("Running streaming Nemo transducer model");
      return std::make_unique<OnlineRecognizerTransducerNeMoImpl>(config);
    }
  }

  if (!config.model_config.paraformer.encoder.empty()) {
    return std::make_unique<OnlineRecognizerParaformerImpl>(config);
  }

  if (!config.model_config.wenet_ctc.model.empty() ||
      !config.model_config.zipformer2_ctc.model.empty() ||
      !config.model_config.nemo_ctc.model.empty()) {
    return std::make_unique<OnlineRecognizerCtcImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  exit(-1);
}

#if __ANDROID_API__ >= 9
std::unique_ptr<OnlineRecognizerImpl> OnlineRecognizerImpl::Create(
    AAssetManager *mgr, const OnlineRecognizerConfig &config) {
  if (!config.model_config.transducer.encoder.empty()) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
    
    auto decoder_model = ReadFile(mgr, config.model_config.transducer.decoder);
    auto sess = std::make_unique<Ort::Session>(env, decoder_model.data(), decoder_model.size(), Ort::SessionOptions{});
    
    size_t node_count = sess->GetOutputCount();
    
    if (node_count == 1) {
      return std::make_unique<OnlineRecognizerTransducerImpl>(mgr, config);
    } else {
      return std::make_unique<OnlineRecognizerTransducerNeMoImpl>(mgr, config);
    }
  }

  if (!config.model_config.paraformer.encoder.empty()) {
    return std::make_unique<OnlineRecognizerParaformerImpl>(mgr, config);
  }

  if (!config.model_config.wenet_ctc.model.empty() ||
      !config.model_config.zipformer2_ctc.model.empty() ||
      !config.model_config.nemo_ctc.model.empty()) {
    return std::make_unique<OnlineRecognizerCtcImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  exit(-1);
}
#endif

}  // namespace sherpa_onnx
