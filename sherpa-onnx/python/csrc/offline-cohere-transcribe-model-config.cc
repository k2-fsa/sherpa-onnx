// sherpa-onnx/python/csrc/offline-cohere-transcribe-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-cohere-transcribe-model-config.h"

#include <string>

#include "sherpa-onnx/python/csrc/offline-cohere-transcribe-model-config.h"

namespace sherpa_onnx {

void PybindOfflineCohereTranscribeModelConfig(py::module *m) {
  using PyClass = OfflineCohereTranscribeModelConfig;
  py::class_<PyClass>(*m, "OfflineCohereTranscribeModelConfig")
      .def(py::init<const std::string &, const std::string &, const std::string &>(),
           py::arg("encoder") = "", py::arg("decoder") = "",
           py::arg("language") = "")
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("language", &PyClass::language)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
