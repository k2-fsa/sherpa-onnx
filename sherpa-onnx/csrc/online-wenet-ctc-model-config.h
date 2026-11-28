// sherpa-onnx/csrc/online-wenet-ctc-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_WENET_CTC_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_WENET_CTC_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OnlineWenetCtcModelConfig {
  std::string model;

  // --chunk_size from wenet
  int32_t chunk_size = 16;

  // --num_left_chunks from wenet
  int32_t num_left_chunks = 4;

  OnlineWenetCtcModelConfig() = default;

  OnlineWenetCtcModelConfig(const std::string &model, int32_t chunk_size,
                            int32_t num_left_chunks)
      : model(model),
        chunk_size(chunk_size),
        num_left_chunks(num_left_chunks) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  // Like Validate(), for files given by a resource manager (Android asset
  // manager / OHOS native resource manager): paths are relative to the
  // resource root.
  template <typename Manager>
  bool Validate(Manager *mgr) const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_WENET_CTC_MODEL_CONFIG_H_
