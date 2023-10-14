// sherpa-onnx/python/csrc/offline-tts-vits-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-tts-vits-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-vits-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsVitsModelConfig(py::module *m) {
  using PyClass = OfflineTtsVitsModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsVitsModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &>(),
           py::arg("model"), py::arg("lexicon"), py::arg("tokens"))
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("lexicon", &PyClass::lexicon)
      .def_readwrite("tokens", &PyClass::tokens)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
