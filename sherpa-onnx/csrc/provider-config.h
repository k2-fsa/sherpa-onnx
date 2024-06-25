// sherpa-onnx/csrc/online-transducer-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONNXRT_EXECUTION_PROVIDER_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONNXRT_EXECUTION_PROVIDER_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/macros.h"
#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

struct CudaConfig {
  uint32_t cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;

  CudaConfig() = default;
  CudaConfig(const uint32_t &cudnn_conv_algo_search)
      : cudnn_conv_algo_search(cudnn_conv_algo_search) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct TensorrtConfig {
  uint32_t trt_max_workspace_size = 2147483648;
  uint32_t trt_max_partition_iterations = 10;
  uint32_t trt_min_subgraph_size = 5;
  bool trt_fp16_enable = 1;
  bool trt_detailed_build_log = 0;
  bool trt_engine_cache_enable = 1;
  bool trt_timing_cache_enable = 1;
  std::string trt_engine_cache_path = ".";
  std::string trt_timing_cache_path = ".";
  bool trt_dump_subgraphs = 0;

  TensorrtConfig() = default;
  TensorrtConfig(const uint32_t &trt_max_workspace_size,
                const uint32_t &trt_max_partition_iterations,
                const uint32_t &trt_min_subgraph_size,
                const bool &trt_fp16_enable,
                const bool &trt_detailed_build_log,
                const bool &trt_engine_cache_enable,
                const bool &trt_timing_cache_enable,
                const std::string &trt_engine_cache_path,
                const std::string &trt_timing_cache_path)
      : trt_max_workspace_size(trt_max_workspace_size),
      trt_max_partition_iterations(trt_max_partition_iterations),
      trt_min_subgraph_size(trt_min_subgraph_size),
      trt_fp16_enable(trt_fp16_enable),
      trt_detailed_build_log(trt_detailed_build_log),
      trt_engine_cache_enable(trt_engine_cache_enable),
      trt_timing_cache_enable(trt_timing_cache_enable),
      trt_engine_cache_path(trt_engine_cache_path),
      trt_timing_cache_path(trt_timing_cache_path) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct ProviderConfig {
  uint32_t device = 0;
  // device only used for cuda and trt
  std::string provider = "cpu";
  CudaConfig cuda_config;
  TensorrtConfig trt_config;

  ProviderConfig() = default;
  ProviderConfig(const uint32_t &device,
                const std::string &provider,
                CudaConfig &cuda_config,
                TensorrtConfig &trt_config)
      : device(device), provider(provider),
        cuda_config(cuda_config),
        trt_config(trt_config) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONNXRT_EXECUTION_PROVIDER_CONFIG_H_
