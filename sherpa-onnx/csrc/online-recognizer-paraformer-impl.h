// sherpa-onnx/csrc/online-recognizer-paraformer-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_PARAFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_PARAFORMER_IMPL_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-lm.h"
#include "sherpa-onnx/csrc/online-paraformer-model.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

class OnlineRecognizerParaformerImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerParaformerImpl(const OnlineRecognizerConfig &config)
      : config_(config),
        model_(OnlineParaformerModel::Create(config.model_config)),
        sym_(config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    else if (config.decoding_method == "greedy_search") {
      // add greedy search decoder
      SHERPA_ONNX_LOGE("to be implemented");
      exit(-1);
    }
    else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      exit(-1);
    }
  }

#if __ANDROID_API__ >= 9
  explicit OnlineRecognizerParaformerImpl(AAssetManager *mgr,
                                          const OnlineRecognizerConfig &config)
      : config_(config),
        model_(OnlineParaformerModel::Create(mgr, config.model_config)),
        sym_(mgr, config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (config.decoding_method == "greedy_search") {
      // add greedy search decoder
      SHERPA_ONNX_LOGE("to be implemented");
      exit(-1);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      exit(-1);
    }
  }
#endif

  void InitOnlineStream(OnlineStream *stream) const override {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
  }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStream>(config_.feat_config);
    InitOnlineStream(stream.get());
    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
    return {};
  }

  bool IsEndpoint(OnlineStream *s) const override {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
    return false;
  }

  void Reset(OnlineStream *s) const override {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
  }

 private:
  OnlineRecognizerConfig config_;
  std::unique_ptr<OnlineParaformerModel> model_;
  SymbolTable sym_;
  Endpoint endpoint_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_PARAFORMER_IMPL_H_
