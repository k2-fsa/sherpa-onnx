// sherpa-onnx/python/csrc/offline-tts-supertonic-model-config.cc
//
// Copyright (c)  2026 zengyw

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
                    const std::string &, const std::string &,
                    const std::string &>(),
           py::arg("duration_predictor"), py::arg("text_encoder"),
           py::arg("vector_estimator"), py::arg("vocoder"), py::arg("tts_json"),
           py::arg("unicode_indexer"), py::arg("voice_style"))
      .def_readwrite("duration_predictor", &PyClass::duration_predictor)
      .def_readwrite("text_encoder", &PyClass::text_encoder)
      .def_readwrite("vector_estimator", &PyClass::vector_estimator)
      .def_readwrite("vocoder", &PyClass::vocoder)
      .def_readwrite("tts_json", &PyClass::tts_json)
      .def_readwrite("unicode_indexer", &PyClass::unicode_indexer)
      .def_readwrite("voice_style", &PyClass::voice_style)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
