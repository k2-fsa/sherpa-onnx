// sherpa-onnx/python/csrc/offline-tts-pocket-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-tts-pocket-model-config.h"

#include "sherpa-onnx/csrc/offline-tts-pocket-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsPocketModelConfig(py::module *m) {
  using PyClass = OfflineTtsPocketModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsPocketModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &>(),
           py::arg("lm_flow"), py::arg("lm_main"), py::arg("encoder"),
           py::arg("decoder"), py::arg("vocab_json"),
           py::arg("token_scores_json"))
      .def_readwrite("lm_flow", &PyClass::lm_flow)
      .def_readwrite("lm_main", &PyClass::lm_main)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("vocab_json", &PyClass::vocab_json)
      .def_readwrite("token_scores_json", &PyClass::token_scores_json)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
