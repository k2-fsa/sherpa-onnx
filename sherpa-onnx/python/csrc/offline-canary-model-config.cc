// sherpa-onnx/python/csrc/offline-canary-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-canary-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/python/csrc/offline-canary-model-config.h"

namespace sherpa_onnx {

void PybindOfflineCanaryModelConfig(py::module *m) {
  using PyClass = OfflineCanaryModelConfig;
  py::class_<PyClass>(*m, "OfflineCanaryModelConfig")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &, bool>(),
           py::arg("encoder") = "", py::arg("decoder") = "",
           py::arg("src_lang") = "", py::arg("tgt_lang") = "",
           py::arg("use_pnc") = true)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("src_lang", &PyClass::src_lang)
      .def_readwrite("tgt_lang", &PyClass::tgt_lang)
      .def_readwrite("use_pnc", &PyClass::use_pnc)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
