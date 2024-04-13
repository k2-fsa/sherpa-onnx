// sherpa-onnx/csrc/offline-punctuation-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-punctuation.h"

namespace sherpa_onnx {

class OfflinePunctuationImpl {
 public:
  virtual ~OfflinePunctuationImpl() = default;

  static std::unique_ptr<OfflinePunctuationImpl> Create(
      const OfflinePunctuationConfig &config);

  virtual std::string AddPunctuation(const std::string &text) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_IMPL_H_
