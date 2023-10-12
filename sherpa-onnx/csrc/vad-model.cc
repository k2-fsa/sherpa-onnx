// sherpa-onnx/csrc/vad-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/vad-model.h"

#include "sherpa-onnx/csrc/silero-vad-model.h"

namespace sherpa_onnx {

std::unique_ptr<VadModel> VadModel::Create(const VadModelConfig &config) {
  // TODO(fangjun): Support other VAD models.
  return std::make_unique<SileroVadModel>(config);
}

#if __ANDROID_API__ >= 9
std::unique_ptr<VadModel> VadModel::Create(AAssetManager *mgr,
                                           const VadModelConfig &config) {
  // TODO(fangjun): Support other VAD models.
  return std::make_unique<SileroVadModel>(mgr, config);
}
#endif

}  // namespace sherpa_onnx
