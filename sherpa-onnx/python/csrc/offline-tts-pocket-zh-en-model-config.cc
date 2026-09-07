// sherpa-onnx/python/csrc/offline-tts-pocket-zh-en-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-tts-pocket-zh-en-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-pocket-zh-en-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsPocketZhEnModelConfig(py::module *m) {
  using PyClass = OfflineTtsPocketZhEnModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsPocketZhEnModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, int32_t>(),
           py::arg("step_model"), py::arg("step_encoder"), py::arg("lexicon"),
           py::arg("voice_embedding_cache_capacity") = 50)
      .def_readwrite("step_model", &PyClass::step_model)
      .def_readwrite("step_encoder", &PyClass::step_encoder)
      .def_readwrite("lexicon", &PyClass::lexicon)
      .def_readwrite("voice_embedding_cache_capacity",
                     &PyClass::voice_embedding_cache_capacity)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx