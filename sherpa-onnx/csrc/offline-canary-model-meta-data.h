// sherpa-onnx/csrc/offline-canary-model-meta-data.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_CANARY_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CANARY_MODEL_META_DATA_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_onnx {

struct OfflineCanaryModelMetaData {
  int32_t vocab_size;
  int32_t subsampling_factor = 8;
  int32_t feat_dim = 120;
  std::string normalize_type;
  std::unordered_map<std::string, int32_t> lang2id;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CANARY_MODEL_META_DATA_H_
