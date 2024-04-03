// sherpa-onnx/python/csrc/online-ctc-fst-decoder-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/online-ctc-fst-decoder-config.h"

#include <string>

#include "sherpa-onnx/csrc/online-ctc-fst-decoder-config.h"

namespace sherpa_onnx {

void PybindOnlineCtcFstDecoderConfig(py::module *m) {
  using PyClass = OnlineCtcFstDecoderConfig;
  py::class_<PyClass>(*m, "OnlineCtcFstDecoderConfig")
      .def(py::init<const std::string &, int32_t>(), py::arg("graph") = "",
           py::arg("max_active") = 3000)
      .def_readwrite("graph", &PyClass::graph)
      .def_readwrite("max_active", &PyClass::max_active)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
