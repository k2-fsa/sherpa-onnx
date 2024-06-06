// sherpa-onnx/csrc/online-transducer-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONNXRT_EXECUTION_PROVIDER_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONNXRT_EXECUTION_PROVIDER_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"
#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

struct OnnxrtCudaConfig {
  uint32_t device = 0;
  uint32_t cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;

  OnnxrtCudaConfig() = default;
  OnnxrtCudaConfig(const uint32_t &device,
                   const uint32_t &cudnn_conv_algo_search)
      : device(device), cudnn_conv_algo_search(cudnn_conv_algo_search) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct OnnxrtTensorrtConfig {
  uint32_t device = 0;
  uint32_t trt_max_workspace_size = 2147483648;
  uint32_t trt_max_partition_iterations = 10;
  uint32_t trt_min_subgraph_size = 5;
  uint32_t trt_fp16_enable = 1;
  uint32_t trt_detailed_build_log = 0;
  uint32_t trt_engine_cache_enable = 1;
  std::string trt_engine_cache_path = ".";
  uint32_t trt_timing_cache_enable = 1;
  std::string trt_timing_cache_path = ".";

  OnnxrtTensorrtConfig() = default;
  OnnxrtTensorrtConfig(const uint32_t &device,
                const uint32_t &trt_max_workspace_size,
                const uint32_t &trt_max_partition_iterations,
                const uint32_t &trt_min_subgraph_size,
                const uint32_t &trt_fp16_enable,
                const uint32_t &trt_detailed_build_log,
                const uint32_t &trt_engine_cache_enable,
                const std::string &trt_engine_cache_path,
                const uint32_t &trt_timing_cache_enable,
                const std::string &trt_timing_cache_path)
      : device(device), trt_max_workspace_size(trt_max_workspace_size),
      trt_max_partition_iterations(trt_max_partition_iterations),
      trt_min_subgraph_size(trt_min_subgraph_size),
      trt_fp16_enable(trt_fp16_enable),
      trt_detailed_build_log(trt_detailed_build_log),
      trt_engine_cache_enable(trt_engine_cache_enable),
      trt_engine_cache_path(trt_engine_cache_path),
      trt_timing_cache_enable(trt_timing_cache_enable),
      trt_timing_cache_path(trt_timing_cache_path) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct OnnxrtExecutionProviderConfig {
  std::string provider = "cpu";
  OnnxrtCudaConfig onnxrtcuda;
  OnnxrtTensorrtConfig onnxrttrtconfig;

  OnnxrtExecutionProviderConfig() = default;
  OnnxrtExecutionProviderConfig(const std::string &provider,
                              const OnnxrtCudaConfig &onnxrtcuda,
                              const OnnxrtTensorrtConfig &onnxrttrtconfig)
      : provider(provider), onnxrtcuda(onnxrtcuda),
        onnxrttrtconfig(onnxrttrtconfig) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONNXRT_EXECUTION_PROVIDER_CONFIG_H_
