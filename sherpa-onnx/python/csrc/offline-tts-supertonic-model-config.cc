// sherpa-onnx/python/csrc/offline-tts-supertonic-model-config.cc
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#include "sherpa-onnx/python/csrc/offline-tts-supertonic-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-supertonic-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsSupertonicModelConfig(py::module *m) {
  using PyClass = OfflineTtsSupertonicModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsSupertonicModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &, int32_t, float>(),
           py::arg("model_dir"), py::arg("voice_style"),
           py::arg("num_steps") = 5, py::arg("speed") = 1.05f)
      .def_readwrite("model_dir", &PyClass::model_dir)
      .def_readwrite("voice_style", &PyClass::voice_style)
      .def_readwrite("num_steps", &PyClass::num_steps)
      .def_readwrite("speed", &PyClass::speed)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
