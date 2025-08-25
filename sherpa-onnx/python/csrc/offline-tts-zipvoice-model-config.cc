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
                    const std::string &, const std::string &, int32_t, float,
                    float, float, float>(),
           py::arg("tokens"), py::arg("text_model"),
           py::arg("flow_matching_model"), py::arg("vocoder"),
           py::arg("data_dir") = "", py::arg("pinyin_dict") = "",
           py::arg("num_steps") = 4, py::arg("feat_scale") = 0.1,
           py::arg("t_shift") = 0.5, py::arg("target_rms") = 0.1,
           py::arg("guidance_scale") = 1.0)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("text_model", &PyClass::text_model)
      .def_readwrite("flow_matching_model", &PyClass::flow_matching_model)
      .def_readwrite("vocoder", &PyClass::vocoder)
      .def_readwrite("data_dir", &PyClass::data_dir)
      .def_readwrite("pinyin_dict", &PyClass::pinyin_dict)
      .def_readwrite("num_steps", &PyClass::num_steps)
      .def_readwrite("feat_scale", &PyClass::feat_scale)
      .def_readwrite("t_shift", &PyClass::t_shift)
      .def_readwrite("target_rms", &PyClass::target_rms)
      .def_readwrite("guidance_scale", &PyClass::guidance_scale)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_onnx
