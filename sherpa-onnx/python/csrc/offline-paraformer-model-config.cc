// sherpa-onnx/python/csrc/offline-paraformer-model-config.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/offline-paraformer-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-paraformer-model-config.h"

namespace sherpa_onnx {

void PybindOfflineParaformerModelConfig(py::module *m) {
  using PyClass = OfflineParaformerModelConfig;
  py::class_<PyClass>(*m, "OfflineParaformerModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
