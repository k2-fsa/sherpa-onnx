// sherpa-onnx/csrc/online-zipformer-meta-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER_META_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER_META_CONFIG_H_

#include <cstdint>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OnlineZipformerMetaConfig {
  std::vector<int32_t> encoder_dims;
  std::vector<int32_t> attention_dims;
  std::vector<int32_t> num_encoder_layers;
  std::vector<int32_t> cnn_module_kernels;
  std::vector<int32_t> left_context_len;

  int32_t T = 0;
  int32_t decode_chunk_len = 0;
  int32_t context_size = 2;

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER_META_CONFIG_H_