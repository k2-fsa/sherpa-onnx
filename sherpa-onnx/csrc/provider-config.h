// sherpa-onnx/csrc/provider-config.h
//
// Copyright (c)  2024  Uniphore (Author: Manickavela)

#ifndef SHERPA_ONNX_CSRC_PROVIDER_CONFIG_H_
#define SHERPA_ONNX_CSRC_PROVIDER_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/macros.h"
#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

struct CudaConfig {
  uint32_t cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;

  CudaConfig() = default;
  explicit CudaConfig(uint32_t cudnn_conv_algo_search)
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
  TensorrtConfig(uint32_t trt_max_workspace_size,
                uint32_t trt_max_partition_iterations,
                uint32_t trt_min_subgraph_size,
                bool trt_fp16_enable,
                bool trt_detailed_build_log,
                bool trt_engine_cache_enable,
                bool trt_timing_cache_enable,
                const std::string &trt_engine_cache_path,
                const std::string &trt_timing_cache_path,
                bool trt_dump_subgraphs)
      : trt_max_workspace_size(trt_max_workspace_size),
      trt_max_partition_iterations(trt_max_partition_iterations),
      trt_min_subgraph_size(trt_min_subgraph_size),
      trt_fp16_enable(trt_fp16_enable),
      trt_detailed_build_log(trt_detailed_build_log),
      trt_engine_cache_enable(trt_engine_cache_enable),
      trt_timing_cache_enable(trt_timing_cache_enable),
      trt_engine_cache_path(trt_engine_cache_path),
      trt_timing_cache_path(trt_timing_cache_path),
      trt_dump_subgraphs(trt_dump_subgraphs) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct ProviderConfig {
  TensorrtConfig trt_config;
  CudaConfig cuda_config;
  std::string provider = "cpu";
  uint32_t device = 0;
  // device only used for cuda and trt

  ProviderConfig() = default;
  ProviderConfig(const TensorrtConfig &trt_config,
                const CudaConfig &cuda_config,
                const std::string &provider,
                uint32_t device)
      : device(device), provider(provider),
        cuda_config(cuda_config),
        trt_config(trt_config) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PROVIDER_CONFIG_H_
