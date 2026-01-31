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
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &, int32_t, float,
                    int32_t, int32_t>(),
           py::arg("duration_predictor"), py::arg("text_encoder"),
           py::arg("vector_estimator"), py::arg("vocoder"),
           py::arg("model_dir"), py::arg("voice_style"),
           py::arg("num_steps") = 5, py::arg("speed") = 1.05f,
           py::arg("max_len_korean") = 120, py::arg("max_len_other") = 300)
      .def_readwrite("duration_predictor", &PyClass::duration_predictor)
      .def_readwrite("text_encoder", &PyClass::text_encoder)
      .def_readwrite("vector_estimator", &PyClass::vector_estimator)
      .def_readwrite("vocoder", &PyClass::vocoder)
      .def_readwrite("model_dir", &PyClass::model_dir)
      .def_readwrite("voice_style", &PyClass::voice_style)
      .def_readwrite("num_steps", &PyClass::num_steps)
      .def_readwrite("speed", &PyClass::speed)
      .def_readwrite("max_len_korean", &PyClass::max_len_korean)
      .def_readwrite("max_len_other", &PyClass::max_len_other)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
