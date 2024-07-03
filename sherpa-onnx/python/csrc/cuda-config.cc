// sherpa-onnx/python/csrc/cuda-config.cc
//
// Copyright (c)  2024  Uniphore (Author: Manickavela A)

#include "sherpa-onnx/python/csrc/cuda-config.h"

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/provider-config.h"

namespace sherpa_onnx {

void PybindCudaConfig(py::module *m) {
  using PyClass = CudaConfig;
  py::class_<PyClass>(*m, "CudaConfig")
      .def(py::init<>())
      .def(py::init<int32_t>(),
           py::arg("cudnn_conv_algo_search") = 1)
      .def_readwrite("cudnn_conv_algo_search", &PyClass::cudnn_conv_algo_search)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
