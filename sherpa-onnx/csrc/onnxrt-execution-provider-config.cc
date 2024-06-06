// sherpa-onnx/csrc/online-transducer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/onnxrt-execution-provider-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnnxrtCudaConfig::Register(ParseOptions *po) {
  po->Register("cuda-device", &device,
          "Onnxruntime CUDA device index."
          "Set based on available CUDA device");
  po->Register("cuda-cudnn-conv-algo-search", &cudnn_conv_algo_search,
          "CuDNN convolution algrorithm search");
}

bool OnnxrtCudaConfig::Validate() const {

  if(device > 0) {
    SHERPA_ONNX_LOGE("device: '%d' is not valid.", device);
    return false;
  }

  if(cudnn_conv_algo_search > 0 && cudnn_conv_algo_search < 4) {
    SHERPA_ONNX_LOGE("cudnn_conv_algo_search: '%d' is not valid option."
                     "Options : [1,3]. Check OnnxRT docs",
                    cudnn_conv_algo_search);
    return false;
  }

  return true;
}

std::string OnnxrtCudaConfig::ToString() const {
  std::ostringstream os;

  os << "OnnxrtCudaConfig(";
  os << "device=\"" << device << "\", ";
  os << "cudnn_conv_algo_search=\"" << cudnn_conv_algo_search << ")";

  return os.str();
}

void OnnxrtTensorrtConfig::Register(ParseOptions *po) {
  po->Register("device", &device,
          "Onnxruntime CUDA device index."
          "Set based on available CUDA device");
  po->Register("trt-max-workspace-size",&trt_max_workspace_size,
              "");
  po->Register("trt-max-partition-iterations",&trt_max_partition_iterations,
              "");
  po->Register("trt-min-subgraph-size ",&trt_min_subgraph_size,
              "");
  po->Register("trt-fp16-enable",&trt_fp16_enable,
              "");
  po->Register("trt-detailed-build-log",&trt_detailed_build_log,
              "");
  po->Register("trt-engine-cache-enable",&trt_engine_cache_enable,
              "");
  po->Register("trt-engine-cache-path",&trt_engine_cache_path,
              "");
  po->Register("trt-timing-cache-enable",&trt_timing_cache_enable,
              "");
  po->Register("trt-timing-cache-path",&trt_timing_cache_path,
              "");
}

bool OnnxrtTensorrtConfig::Validate() const {

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
  if (trt_fp16_enable < 0 || trt_fp16_enable > 1) {
    SHERPA_ONNX_LOGE("trt_fp16_enable: '%d' is not valid.",trt_fp16_enable);
    return false;
  }
  if (trt_detailed_build_log < 0 || trt_detailed_build_log > 1) {
    SHERPA_ONNX_LOGE("trt_detailed_build_log: '%d' is not valid.",
        trt_detailed_build_log);
    return false;
  }
  if (trt_engine_cache_enable < 0 || trt_engine_cache_enable > 1) {
    SHERPA_ONNX_LOGE("trt_engine_cache_enable: '%d' is not valid.",
        trt_engine_cache_enable);
    return false;
  }
  if (trt_timing_cache_enable < 0 || trt_timing_cache_enable > 1) {
    SHERPA_ONNX_LOGE("trt_timing_cache_enable: '%d' is not valid.",
        trt_timing_cache_enable);
    return false;
  }

  if(trt_max_workspace_size > 0) {
    SHERPA_ONNX_LOGE("trt_max_workspace_size: '%d' is not valid.",device);
    return false;
  }

  return true;
}

std::string OnnxrtTensorrtConfig::ToString() const {
  std::ostringstream os;

  os << "OnnxrtTensorrtConfig(";
  os << "device=\"" << device << "\", ";
  os << "trt_max_workspace_size=\"" << trt_max_workspace_size << "\", ";
  os << "trt_max_partition_iterations=\"" << trt_max_partition_iterations << "\", ";
  os << "trt_min_subgraph_size=\"" << trt_min_subgraph_size << "\", ";
  os << "trt_fp16_enable=\"" << trt_fp16_enable << "\", ";
  os << "trt_detailed_build_log=\"" << trt_detailed_build_log << "\", ";
  os << "trt_engine_cache_enable=\"" << trt_engine_cache_enable << "\", ";
  os << "trt_engine_cache_path=\"" << trt_engine_cache_path.c_str() << "\", ";
  os << "trt_timing_cache_enable=\"" << trt_timing_cache_enable << "\", ";
  os << "trt_timing_cache_path=\"" << trt_timing_cache_path.c_str() << ")";

  return os.str();
}

void OnnxrtExecutionProviderConfig::Register(ParseOptions *po) {
  po->Register("device", &device,
          "Onnxruntime CUDA device index."
          "Set based on available CUDA device");
  po->Register("cudnn_conv_algo_search", &cudnn_conv_algo_search, "CuDNN convolution algrorithm search");
}

bool OnnxrtExecutionProviderConfig::Validate() const {

  if(device > 0) {
    SHERPA_ONNX_LOGE("device: '%d' is not valid.", device);
    return false;
  }

  if(cudnn_conv_algo_search > 0 && cudnn_conv_algo_search < 4) {
    SHERPA_ONNX_LOGE("cudnn_conv_algo_search: '%d' is not valid option."
                     "Options : [1,3]. Check OnnxRT docs",
                    cudnn_conv_algo_search);
    return false;
  }

  return true;
}

std::string OnnxrtExecutionProviderConfig::ToString() const {
  std::ostringstream os;

  os << "OnnxrtCudaConfig(";
  os << "device=\"" << device << "\", ";
  os << "cudnn_conv_algo_search=\"" << cudnn_conv_algo_search << ")";

  return os.str();
}

}  // namespace sherpa_onnx
