// sherpa-onnx/python/csrc/online-nemo-ctc-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/online-nemo-ctc-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/online-nemo-ctc-model-config.h"

namespace sherpa_onnx {

void PybindOnlineNeMoCtcModelConfig(py::module *m) {
  using PyClass = OnlineNeMoCtcModelConfig;
  py::class_<PyClass>(*m, "OnlineNeMoCtcModelConfig")
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
