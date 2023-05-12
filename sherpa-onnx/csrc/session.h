// sherpa-onnx/csrc/session.h
//
// Copyright (c)  2023  Xiaomi Corporation

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model-config.h"

namespace sherpa_onnx {

Ort::SessionOptions GetSessionOptions(
    const OnlineTransducerModelConfig &config);

Ort::SessionOptions GetSessionOptions(const OfflineModelConfig &config);

}  // namespace sherpa_onnx
