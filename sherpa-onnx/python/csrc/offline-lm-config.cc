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
      .def(py::init<const std::string &, float>(), py::arg("model"),
           py::arg("scale"))
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("scale", &PyClass::scale)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
