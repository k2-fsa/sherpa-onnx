// sherpa-onnx/csrc/session.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SESSION_H_
#define SHERPA_ONNX_CSRC_SESSION_H_

#include <string>
#include <unordered_map>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-lm-config.h"
#include "sherpa-onnx/csrc/online-lm-config.h"
#include "sherpa-onnx/csrc/online-model-config.h"

namespace sherpa_onnx {

// Build the key/value pairs forwarded to the CUDA execution provider via
// OrtApi::UpdateCUDAProviderOptions, e.g., gpu_mem_limit or
// arena_extend_strategy. Keys in |config| take precedence; |device_id| and
// |cudnn_conv_algo_search| supply defaults only when the corresponding keys
// are absent. "DEBUG" is consumed by the config-file parser and dropped here.
// Exposed for testing.
std::unordered_map<std::string, std::string> BuildCudaProviderOptions(
    std::unordered_map<std::string, std::string> config, int32_t device_id,
    OrtCudnnConvAlgoSearch cudnn_conv_algo_search);

Ort::SessionOptions GetSessionOptionsImpl(
    int32_t num_threads, const std::string &provider_str,
    const ProviderConfig *provider_config = nullptr);

Ort::SessionOptions GetSessionOptions(const OfflineLMConfig &config);
Ort::SessionOptions GetSessionOptions(const OnlineLMConfig &config);

Ort::SessionOptions GetSessionOptions(const OnlineModelConfig &config);

Ort::SessionOptions GetSessionOptions(const OnlineModelConfig &config,
                                      const std::string &model_type);

Ort::SessionOptions GetSessionOptions(int32_t num_threads,
                                      const std::string &provider_str);

template <typename T>
Ort::SessionOptions GetSessionOptions(const T &config) {
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SESSION_H_
