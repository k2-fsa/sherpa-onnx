// sherpa-onnx/python/csrc/offline-fire-red-asr-ctc-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-fire-red-asr-ctc-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/python/csrc/offline-fire-red-asr-ctc-model-config.h"

namespace sherpa_onnx {

void PybindOfflineFireRedAsrCtcModelConfig(py::module *m) {
  using PyClass = OfflineFireRedAsrCtcModelConfig;
  py::class_<PyClass>(*m, "OfflineFireRedAsrCtcModelConfig")
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
