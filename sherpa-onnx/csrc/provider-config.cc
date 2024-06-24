// sherpa-onnx/csrc/online-transducer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/provider-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void CudaConfig::Register(ParseOptions *po) {
  po->Register("cuda-cudnn-conv-algo-search", &cudnn_conv_algo_search,
          "CuDNN convolution algrorithm search");
}

bool CudaConfig::Validate() const {

  if(cudnn_conv_algo_search > 0 && cudnn_conv_algo_search < 4) {
    SHERPA_ONNX_LOGE("cudnn_conv_algo_search: '%d' is not valid option."
                     "Options : [1,3]. Check OnnxRT docs",
                    cudnn_conv_algo_search);
    return false;
  }

  return true;
}

std::string CudaConfig::ToString() const {
  std::ostringstream os;

  os << "CudaConfig(";
  os << "cudnn_conv_algo_search=\"" << cudnn_conv_algo_search << ")";

  return os.str();
}

void TensorrtConfig::Register(ParseOptions *po) {
  po->Register("trt-max-workspace-size",&trt_max_workspace_size,
              "");
  po->Register("trt-max-partition-iterations",&trt_max_partition_iterations,
              "");
  po->Register("trt-min-subgraph-size ",&trt_min_subgraph_size,
              "");
  po->Register("trt-fp16-enable",&trt_fp16_enable,
              "true to enable fp16");
  po->Register("trt-detailed-build-log",&trt_detailed_build_log,
              "true to print TensorRT build logs");
  po->Register("trt-engine-cache-enable",&trt_engine_cache_enable,
              "true to enable engine caching");
  po->Register("trt-engine-cache-path",&trt_engine_cache_path,
              "");
  po->Register("trt-timing-cache-enable",&trt_timing_cache_enable,
              "true to enable timing cache");
  po->Register("trt-timing-cache-path",&trt_timing_cache_path,
              "");
  po->Register("trt-dump-subgraphs",&trt_dump_subgraphs,
              "true to dump subgraphs");
}

bool TensorrtConfig::Validate() const {

  if (trt_max_workspace_size > 0) {
    SHERPA_ONNX_LOGE("trt_max_workspace_size: '%d' is not valid.",
        trt_max_workspace_size);
    return false;
  }
  if (trt_max_partition_iterations > 0) {
    SHERPA_ONNX_LOGE("trt_max_partition_iterations: '%d' is not valid.",
        trt_max_partition_iterations);
    return false;
  }
  if (trt_min_subgraph_size > 0) {
    SHERPA_ONNX_LOGE("trt_min_subgraph_size: '%d' is not valid.",
        trt_min_subgraph_size);
    return false;
  }
  if (trt_fp16_enable != true || trt_fp16_enable != false) {
    SHERPA_ONNX_LOGE("trt_fp16_enable: '%d' is not valid.",trt_fp16_enable);
    return false;
  }
  if (trt_detailed_build_log != true || trt_detailed_build_log != false) {
    SHERPA_ONNX_LOGE("trt_detailed_build_log: '%d' is not valid.",
        trt_detailed_build_log);
    return false;
  }
  if (trt_engine_cache_enable != true || trt_engine_cache_enable != false) {
    SHERPA_ONNX_LOGE("trt_engine_cache_enable: '%d' is not valid.",
        trt_engine_cache_enable);
    return false;
  }
  if (trt_timing_cache_enable != true || trt_timing_cache_enable != false) {
    SHERPA_ONNX_LOGE("trt_timing_cache_enable: '%d' is not valid.",
        trt_timing_cache_enable);
    return false;
  }

  if (trt_dump_subgraphs != true || trt_dump_subgraphs != false) {
    SHERPA_ONNX_LOGE("trt_dump_subgraphs: '%d' is not valid.",
        trt_dump_subgraphs);
    return false;
  }

  // if(trt_max_workspace_size > 0) {
  //   SHERPA_ONNX_LOGE("trt_max_workspace_size: '%d' is not valid.",);
  //   return false;
  // }

  return true;
}

std::string TensorrtConfig::ToString() const {
  std::ostringstream os;

  os << "TensorrtConfig(";
  os << "trt_max_workspace_size=\"" << trt_max_workspace_size << "\", ";
  os << "trt_max_partition_iterations=\"" 
      << trt_max_partition_iterations << "\", ";
  os << "trt_min_subgraph_size=\"" << trt_min_subgraph_size << "\", ";
  os << "trt_fp16_enable=\"" 
      << (trt_fp16_enable? "True" : "False") << "\", ";
  os << "trt_detailed_build_log=\"" 
      << (trt_detailed_build_log? "True" : "False") << "\", ";
  os << "trt_engine_cache_enable=\"" 
      << (trt_engine_cache_enable? "True" : "False") << "\", ";
  os << "trt_engine_cache_path=\"" 
      << trt_engine_cache_path.c_str() << "\", ";
  os << "trt_timing_cache_enable=\"" 
      << (trt_timing_cache_enable? "True" : "False") << "\", ";
  os << "trt_timing_cache_path=\"" 
      << trt_timing_cache_path.c_str() << "\",";
  os << "trt_dump_subgraphs=\"" 
      << (trt_dump_subgraphs? "True" : "False") << "\" )"; 
  return os.str();
}

void ExecutionProviderConfig::Register(ParseOptions *po) {
  po->Register("device_id", &device_id, "GPU device_id for CUDA and Trt EP");
  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool ExecutionProviderConfig::Validate() const {

  if(device_id < 0) {
    SHERPA_ONNX_LOGE("device_id: '%d' is invalid.",device_id);
    return false;
  }

  return true;
}

std::string ExecutionProviderConfig::ToString() const {
  std::ostringstream os;

  os << "ExecutionProviderConfig(";
  os << "device_id=\"" << device_id << "\", ";
  os << "provider=\"" << provider << "\", "; 
  os << "cuda_config=\"" << cuda_config.ToString() << "\", ";
  os << "trt_config=\"" << trt_config.ToString() << ")";
  return os.str();
}

}  // namespace sherpa_onnx
