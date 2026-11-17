// sherpa-onnx/csrc/session.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SESSION_H_
#define SHERPA_ONNX_CSRC_SESSION_H_

#include <cstdint>
#include <string>
#include <unordered_map>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-lm-config.h"
#include "sherpa-onnx/csrc/online-lm-config.h"
#include "sherpa-onnx/csrc/online-model-config.h"

namespace sherpa_onnx {

// The CUDA EP was successfully appended to the session options. This does not
// imply that every node in a subsequently loaded model will execute on CUDA.
struct CudaProviderInfo {
  bool enabled = false;
  int32_t device_id = 0;  // Only meaningful when enabled is true.
};

// Build the key/value pairs forwarded to the CUDA execution provider via
// OrtApi::UpdateCUDAProviderOptions, e.g., gpu_mem_limit or
// arena_extend_strategy. Keys in |config| take precedence; |device_id| and
// |cudnn_conv_algo_search| supply defaults only when the corresponding keys
// are absent. "DEBUG" is consumed by the config-file parser and dropped here.
// Exposed for testing.
std::unordered_map<std::string, std::string> BuildCudaProviderOptions(
    std::unordered_map<std::string, std::string> config, int32_t device_id,
    OrtCudnnConvAlgoSearch cudnn_conv_algo_search);

// Read the device from the final CUDA options. Reject invalid/negative IDs
// without changing *device_id. Exposed for testing.
bool GetCudaDeviceId(
    const std::unordered_map<std::string, std::string> &options,
    int32_t *device_id);

Ort::SessionOptions GetSessionOptionsImpl(
    int32_t num_threads, const std::string &provider_str,
    const ProviderConfig *provider_config = nullptr);

Ort::SessionOptions GetSessionOptionsImpl(
    int32_t num_threads, const std::string &provider_str,
    const ProviderConfig *provider_config,
    CudaProviderInfo *cuda_provider_info);

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

template <typename T>
Ort::SessionOptions GetSessionOptions(const T &config,
                                     CudaProviderInfo *cuda_provider_info) {
  return GetSessionOptionsImpl(config.num_threads, config.provider, nullptr,
                               cuda_provider_info);
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SESSION_H_
