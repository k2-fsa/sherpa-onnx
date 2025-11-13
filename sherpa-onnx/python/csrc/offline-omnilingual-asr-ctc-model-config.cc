// sherpa-onnx/python/csrc/offline-omnilingual-asr-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-omnilingual-asr-ctc-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-omnilingual-asr-ctc-model-config.h"

namespace sherpa_onnx {

void PybindOfflineOmnilingualAsrCtcModelConfig(py::module *m) {
  using PyClass = OfflineOminlingualAsrCtcModelConfig;
  py::class_<PyClass>(*m, "OfflineOminlingualAsrCtcModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
