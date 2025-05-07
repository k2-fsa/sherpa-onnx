// sherpa-onnx/csrc/offline-dolphin-model-meta-data.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_MODEL_META_DATA_H_

#include <string>
#include <vector>

namespace sherpa_onnx {

struct OfflineDolphinModelMetaData {
  int32_t vocab_size;
  int32_t subsampling_factor = 4;
  std::vector<float> mean;
  std::vector<float> inv_stddev;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_MODEL_META_DATA_H_
