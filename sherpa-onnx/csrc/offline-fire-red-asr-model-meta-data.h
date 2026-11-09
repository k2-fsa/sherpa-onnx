// sherpa-onnx/csrc/offline-fire-red-asr-model-meta-data.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_MODEL_META_DATA_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_onnx {

struct OfflineFireRedAsrModelMetaData {
  int32_t sos_id = 0;
  int32_t eos_id = 0;
  int32_t max_len = 0;

  int32_t num_decoder_layers = 0;
  int32_t num_head = 0;
  int32_t head_dim = 0;

  std::vector<float> mean;
  std::vector<float> inv_stddev;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_MODEL_META_DATA_H_
