// sherpa-onnx/python/csrc/provider-config.cc
//
// Copyright (c)  2024  Uniphore (Author: Manickavela A)


#include "sherpa-onnx/python/csrc/provider-config.h"

#include <string>

#include "sherpa-onnx/csrc/provider-config.h"

namespace sherpa_onnx {

static void PybindCudaConfig(py::module *m) {
  using PyClass = CudaConfig;
  py::class_<PyClass>(*m, "CudaConfig")
      .def(py::init<int32_t>(),
           py::arg("cudnn_conv_algo_search") = 1)
      .def_readwrite("cudnn_conv_algo_search", &PyClass::cudnn_conv_algo_search)
      .def("__str__", &PyClass::ToString);
}

static void PybindTensorrtConfig(py::module *m) {
  using PyClass = TensorrtConfig;
  py::class_<PyClass>(*m, "TensorrtConfig")
      .def(py::init<int32_t, int32_t, int32_t, bool,
          bool, bool, bool, const std::string &,
          const std::string &, bool>(),
           py::arg("trt_max_workspace_size") = 2147483648,
           py::arg("trt_max_partition_iterations") = 10,
           py::arg("trt_min_subgraph_size") = 5,
           py::arg("trt_fp16_enable") = true,
           py::arg("trt_detailed_build_log") = false,
           py::arg("trt_engine_cache_enable") = true,
           py::arg("trt_timing_cache_enable") = true,
           py::arg("trt_engine_cache_path") = ".",
           py::arg("trt_timing_cache_path") = ".",
           py::arg("trt_dump_subgraphs") = false)
      .def_readwrite("trt_max_workspace_size", &PyClass::trt_max_workspace_size)
      .def_readwrite("trt_max_partition_iterations", &PyClass::trt_max_partition_iterations)
      .def_readwrite("trt_min_subgraph_size", &PyClass::trt_min_subgraph_size)
      .def_readwrite("trt_fp16_enable", &PyClass::trt_fp16_enable)
      .def_readwrite("trt_detailed_build_log", &PyClass::trt_detailed_build_log)
      .def_readwrite("trt_engine_cache_enable", &PyClass::trt_engine_cache_enable)
      .def_readwrite("trt_timing_cache_enable", &PyClass::trt_timing_cache_enable)
      .def_readwrite("trt_engine_cache_path", &PyClass::trt_engine_cache_path)
      .def_readwrite("trt_timing_cache_path", &PyClass::trt_timing_cache_path)
      .def_readwrite("trt_dump_subgraphs", &PyClass::trt_dump_subgraphs)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

void PybindProviderConfig(py::module *m) {
  PybindCudaConfig(m);
  PybindTensorrtConfig(m);

  using PyClass = ProviderConfig;
  py::class_<PyClass>(*m, "ProviderConfig")
      .def(py::init<const TensorrtConfig &,
          const CudaConfig &, const std::string &,
          int32_t>(),
           py::arg("trt_config") = TensorrtConfig(),
           py::arg("cuda_config") = CudaConfig(),
           py::arg("provider") = "cpu",
           py::arg("device") = 0)
      .def_readwrite("trt_config", &PyClass::trt_config)
      .def_readwrite("cuda_config", &PyClass::cuda_config)
      .def_readwrite("provider", &PyClass::provider)
      .def_readwrite("device", &PyClass::device)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
