// sherpa-onnx/python/csrc/offline-zipformer-ctc-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-zipformer-ctc-model-config.h"

#include "sherpa-onnx/csrc/offline-zipformer-ctc-model-config.h"

namespace sherpa_onnx {

void PybindOfflineZipformerCtcModelConfig(py::module *m) {
  using PyClass = OfflineZipformerCtcModelConfig;
  py::class_<PyClass>(*m, "OfflineZipformerCtcModelConfig")
      .def(py::init<const std::string &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
