// sherpa-onnx/python/csrc/online-lm-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/online-lm-config.h"

#include <string>

#include "sherpa-onnx//csrc/online-lm-config.h"

namespace sherpa_onnx {

void PybindOnlineLMConfig(py::module *m) {
  using PyClass = OnlineLMConfig;
  py::class_<PyClass>(*m, "OnlineLMConfig")
      .def(py::init<const std::string &, float>(), py::arg("model") = "",
           py::arg("scale") = 0.5f)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("scale", &PyClass::scale)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
