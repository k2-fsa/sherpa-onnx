// sherpa-onnx/python/csrc/offline-fire-red-asr-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-fire-red-asr-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/python/csrc/offline-fire-red-asr-model-config.h"

namespace sherpa_onnx {

void PybindOfflineFireRedAsrModelConfig(py::module *m) {
  using PyClass = OfflineFireRedAsrModelConfig;
  py::class_<PyClass>(*m, "OfflineFireRedAsrModelConfig")
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("encoder"), py::arg("decoder"))
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
