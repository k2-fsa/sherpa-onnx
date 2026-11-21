// sherpa-onnx/python/csrc/offline-w2v-bert-ctc-model-config.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-w2v-bert-ctc-model-config.h"

#include <string>

#include "sherpa-onnx/python/csrc/offline-w2v-bert-ctc-model-config.h"

namespace sherpa_onnx {

void PybindOfflineW2vBertCtcModelConfig(py::module *m) {
  using PyClass = OfflineW2vBertCtcModelConfig;
  py::class_<PyClass>(*m, "OfflineW2vBertCtcModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
