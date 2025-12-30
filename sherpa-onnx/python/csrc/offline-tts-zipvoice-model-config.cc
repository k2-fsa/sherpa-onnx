// sherpa-onnx/python/csrc/offline-tts-zipvoice-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-tts-zipvoice-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-zipvoice-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTtsZipvoiceModelConfig(py::module *m) {
  using PyClass = OfflineTtsZipvoiceModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsZipvoiceModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &, float, float,
                    float, float>(),
           py::arg("tokens"), py::arg("encoder"), py::arg("decoder"),
           py::arg("vocoder"), py::arg("data_dir") = "",
           py::arg("lexicon") = "", py::arg("feat_scale") = 0.1,
           py::arg("t_shift") = 0.5, py::arg("target_rms") = 0.1,
           py::arg("guidance_scale") = 1.0)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("vocoder", &PyClass::vocoder)
      .def_readwrite("data_dir", &PyClass::data_dir)
      .def_readwrite("lexicon", &PyClass::lexicon)
      .def_readwrite("feat_scale", &PyClass::feat_scale)
      .def_readwrite("t_shift", &PyClass::t_shift)
      .def_readwrite("target_rms", &PyClass::target_rms)
      .def_readwrite("guidance_scale", &PyClass::guidance_scale)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
