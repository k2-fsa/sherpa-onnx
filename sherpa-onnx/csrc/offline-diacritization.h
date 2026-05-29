// sherpa-onnx/csrc/offline-diacritization.h
//
// Copyright (c)  2026  Matias Lin
#ifndef SHERPA_ONNX_CSRC_OFFLINE_DIACRITIZATION_H_
#define SHERPA_ONNX_CSRC_OFFLINE_DIACRITIZATION_H_

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/offline-diacritization-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineDiacritizationConfig {
  OfflineDiacritizationModelConfig model;

  OfflineDiacritizationConfig() = default;

  explicit OfflineDiacritizationConfig(
      const OfflineDiacritizationModelConfig &model)
      : model(model) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class OfflineDiacritizationImpl;

class OfflineDiacritization {
 public:
  explicit OfflineDiacritization(const OfflineDiacritizationConfig &config);

  template <typename Manager>
  OfflineDiacritization(Manager *mgr,
                        const OfflineDiacritizationConfig &config);

  ~OfflineDiacritization();

  // Add diacritics to the input text and return it
  std::string AddDiacritics(const std::string &text) const;

 private:
  std::unique_ptr<OfflineDiacritizationImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_DIACRITIZATION_H_
