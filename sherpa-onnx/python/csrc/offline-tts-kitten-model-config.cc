// sherpa-onnx/python/csrc/offline-tts-kitten-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-tts-kitten-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-kitten-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsKittenModelConfig(py::module *m) {
  using PyClass = OfflineTtsKittenModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsKittenModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &, float,
                    const std::string &>(),
           py::arg("model"), py::arg("voices"), py::arg("tokens"),
           py::arg("data_dir") = "", py::arg("length_scale") = 1.0,
           py::arg("lexicon") = "")
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("voices", &PyClass::voices)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("lexicon", &PyClass::lexicon)
      .def_readwrite("data_dir", &PyClass::data_dir,
                     "Deprecated. Ignored since v2.0.0: espeak-ng is no longer used. Use lexicon, or GenerationConfig.phoneme_codepoints / tokens with an external phonemizer instead.")
      .def_readwrite("length_scale", &PyClass::length_scale)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
