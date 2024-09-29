// sherpa-onnx/csrc/fast-clustering-config.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FAST_CLUSTERING_CONFIG_H_
#define SHERPA_ONNX_CSRC_FAST_CLUSTERING_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct FastClusteringConfig {
  // If greater than 0, then threshold is ignored
  int32_t num_clusters = -1;

  // distance threshold
  float threshold = 0.5;

  std::string ToString() const;

  void Register(ParseOptions *po);
  bool Validate() const;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_FAST_CLUSTERING_CONFIG_H_
