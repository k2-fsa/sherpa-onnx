// sherpa-onnx/python/csrc/offline-source-separation-spleeter-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-source-separation-spleeter-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-source-separation-spleeter-model-config.h"

namespace sherpa_onnx {

void PybindOfflineSourceSeparationSpleeterModelConfig(py::module *m) {
  using PyClass = OfflineSourceSeparationSpleeterModelConfig;
  py::class_<PyClass>(*m, "OfflineSourceSeparationSpleeterModelConfig")
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("vocals") = "", py::arg("accompaniment") = "")
      .def_readwrite("vocals", &PyClass::vocals)
      .def_readwrite("accompaniment", &PyClass::accompaniment)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
