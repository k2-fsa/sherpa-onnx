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
      .def(py::init<const std::string &, const std::string &, const std::string &,
                    bool, bool>(),
           py::arg("encoder") = "", py::arg("decoder") = "",
           py::arg("language") = "", py::arg("use_punct") = true,
           py::arg("use_itn") = true)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("language", &PyClass::language)
      .def_readwrite("use_punct", &PyClass::use_punct)
      .def_readwrite("use_itn", &PyClass::use_itn)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
