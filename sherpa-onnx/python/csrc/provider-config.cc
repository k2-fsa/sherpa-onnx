// sherpa-onnx/python/csrc/provider-config.h
//
// Copyright (c)  2024 Uniphore Pvt Ltd(github.com/manickavela29)


#include "sherpa-onnx/csrc/provider-config.h"

#include <string>

#include "sherpa-onnx/python/csrc/provider-config.h"

namespace sherpa_onnx {

void PybindCudaConfig(py::module *m) {
  using PyClass = PybindCudaConfig;
  py::class_<PyClass>(*m, "PybindCudaConfig")
      .def(py::init<const uint32_t cudnn_conv_algo_search &,
           py::arg("cudnn_conv_algo_search") = 1)
      .def_readwrite("cudnn_conv_algo_search", &PyClass::cudnn_conv_algo_search)
      .def("__str__", &PyClass::ToString);
}

void PybindTensorrtConfig(py::module *m) {
  using PyClass = PybindTensorrtConfig;
  py::class_<PyClass>(*m, "PybindTensorrtConfig")
      .def(py::init<const TensorrtConfig &, const CudaConfig &,
                    const std::string &, const uint32_t &>(),
           py::arg("trt_config") = TensorrtConfig(),
           py::arg("cuda_config") = CudaConfig(),
           py::arg("provider") = "cpu",
           py::arg("device") = 0)
      .def_readwrite("trt_config", &PyClass::Ten)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("joiner", &PyClass::joiner)
      .def("__str__", &PyClass::ToString);
}


void PybindProviderConfig(py::module *m) {
  using PyClass = ProviderConfig;
  py::class_<PyClass>(*m, "ProviderConfig")
      .def(py::init<const TensorrtConfig &, const CudaConfig &,
                    const std::string &, const uint32_t &>(),
           py::arg("trt_config") = TensorrtConfig(),
           py::arg("cuda_config") = CudaConfig(),
           py::arg("provider") = "cpu",
           py::arg("device") = 0)
      .def_readwrite("trt_config", &PyClass::Ten)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("joiner", &PyClass::joiner)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
