// sherpa-onnx/python/csrc/offline-wenet-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-wenet-ctc-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/python/csrc/offline-wenet-ctc-model-config.h"

namespace sherpa_onnx {

void PybindOfflineWenetCtcModelConfig(py::module *m) {
  using PyClass = OfflineWenetCtcModelConfig;
  py::class_<PyClass>(*m, "OfflineWenetCtcModelConfig")
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
