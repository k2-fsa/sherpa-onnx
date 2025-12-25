// sherpa-onnx/python/csrc/offline-medasr-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-medasr-ctc-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/python/csrc/offline-medasr-ctc-model-config.h"

namespace sherpa_onnx {

void PybindOfflineMedAsrCtcModelConfig(py::module *m) {
  using PyClass = OfflineMedAsrCtcModelConfig;
  py::class_<PyClass>(*m, "OfflineMedAsrCtcModelConfig")
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
