// sherpa-onnx/python/csrc/offline-lm-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-lm-config.h"

#include <string>

#include "sherpa-onnx//csrc/offline-lm-config.h"

namespace sherpa_onnx {

void PybindOfflineLMConfig(py::module *m) {
  using PyClass = OfflineLMConfig;
  py::class_<PyClass>(*m, "OfflineLMConfig")
      .def(py::init<const std::string &, float, int32_t, const std::string &>(),
           py::arg("model"), py::arg("scale") = 0.5f,
           py::arg("lm_num_threads") = 1, py::arg("lm-provider") = "cpu")
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("scale", &PyClass::scale)
      .def_readwrite("lm_provider", &PyClass::lm_provider)
      .def_readwrite("lm_num_threads", &PyClass::lm_num_threads)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
