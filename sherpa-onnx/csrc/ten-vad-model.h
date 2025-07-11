// sherpa-onnx/csrc/ten-vad-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_TEN_VAD_MODEL_H_
#define SHERPA_ONNX_CSRC_TEN_VAD_MODEL_H_

#include <memory>

#include "sherpa-onnx/csrc/vad-model.h"

namespace sherpa_onnx {
class TenVadModel : public VadModel {
 public:
  explicit TenVadModel(const VadModelConfig &config);
}

#endif  // SHERPA_ONNX_CSRC_TEN_VAD_MODEL_H_
