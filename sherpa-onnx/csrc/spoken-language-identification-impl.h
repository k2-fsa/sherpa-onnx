// sherpa-onnx/csrc/spoken-language-identification-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_IMPL_H_
#define SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_IMPL_H_

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/spoken-language-identification.h"

namespace sherpa_onnx {

class SpokenLanguageIdentificationImpl {
 public:
  virtual ~SpokenLanguageIdentificationImpl() = default;

  static std::unique_ptr<SpokenLanguageIdentificationImpl> Create(
      const SpokenLanguageIdentificationConfig &config);

  virtual std::unique_ptr<OfflineStream> CreateStream() const = 0;

  virtual std::string Compute(OfflineStream *s) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_IMPL_H_
