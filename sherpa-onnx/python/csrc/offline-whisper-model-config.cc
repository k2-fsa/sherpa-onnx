// sherpa-onnx/python/csrc/offline-whisper-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/python/csrc/offline-whisper-model-config.h"

namespace sherpa_onnx {

void PybindOfflineWhisperModelConfig(py::module *m) {
  using PyClass = OfflineWhisperModelConfig;
  py::class_<PyClass>(*m, "OfflineWhisperModelConfig")
      .def(py::init<const std::string &, const std::string &>(),
           py::arg("encoder"), py::arg("decoder"))
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
