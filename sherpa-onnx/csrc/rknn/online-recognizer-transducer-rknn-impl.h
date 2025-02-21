// sherpa-onnx/csrc/rknn/online-recognizer-transducer-rknn-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RKNN_ONLINE_RECOGNIZER_TRANSDUCER_RKNN_IMPL_H_
#define SHERPA_ONNX_CSRC_RKNN_ONLINE_RECOGNIZER_TRANSDUCER_RKNN_IMPL_H_

#include <algorithm>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-lm.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

class OnlineRecognizerTransducerRknnImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerTransducerRknnImpl(
      const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        endpoint_(config_.endpoint_config),
        model_(std::make_unique<OnlineZipformerTransducerModelRknn>(
            config.model_config)) {
    exit(10);
  }

  template <typename Manager>
  explicit OnlineRecognizerTransducerRknnImpl(
      Manager *mgr, const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        endpoint_(config_.endpoint_config),
        model_(std::make_unique<OnlineZipformerTransducerModelRknn>(mgr,
                                                                    config)) {}

  std::unique_ptr<OnlineStream> CreateStream() const override { return {}; }

  std::unique_ptr<OnlineStream> CreateStream(
      const std::string &hotwords) const override {
    return {};
  }

  bool IsReady(OnlineStream *s) const override { return false; }

  // Warmping up engine with wp: warm_up count and max-batch-size

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {}

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    return {};
  }

  bool IsEndpoint(OnlineStream *s) const override { return false; }

  void Reset(OnlineStream *s) const override {}

 private:
  OnlineRecognizerConfig config_;
  SymbolTable sym_;
  Endpoint endpoint_;
  int32_t unk_id_ = -1;
  std::unique_ptr<OnlineZipformerTransducerModelRknn> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_ONLINE_RECOGNIZER_TRANSDUCER_RKNN_IMPL_H_
