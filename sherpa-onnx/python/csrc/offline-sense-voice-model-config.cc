// sherpa-onnx/python/csrc/offline-sense-voice-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-sense-voice-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/python/csrc/offline-sense-voice-model-config.h"

namespace sherpa_onnx {

void PybindOfflineSenseVoiceModelConfig(py::module *m) {
  using PyClass = OfflineSenseVoiceModelConfig;
  py::class_<PyClass>(*m, "OfflineSenseVoiceModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &, bool>(),
           py::arg("model"), py::arg("language"), py::arg("use_itn"))
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("language", &PyClass::language)
      .def_readwrite("use_itn", &PyClass::use_itn)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
