// sherpa-onnx/csrc/offline-punctuation-ct-transformer-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_CT_TRANSFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_CT_TRANSFORMER_IMPL_H_

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/audio-tagging-impl.h"
#include "sherpa-onnx/csrc/audio-tagging-label-file.h"
#include "sherpa-onnx/csrc/audio-tagging.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-zipformer-audio-tagging-model.h"

namespace sherpa_onnx {

class OfflinePunctuationCtTransformerImpl : public OfflinePunctuationImpl {
 public:
  explicit OfflinePunctuationCtTransformerImpl(
      const OfflinePunctuationConfig &config)
      : config_(config) {}

  std::string AddPunctuation(const std::string &text) const override {
    return {};
  }

 private:
  OfflinePunctuationConfig config_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_CT_TRANSFORMER_IMPL_H_
