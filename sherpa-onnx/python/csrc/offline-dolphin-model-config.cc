// sherpa-onnx/python/csrc/offline-dolphin-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-dolphin-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/python/csrc/offline-dolphin-model-config.h"

namespace sherpa_onnx {

void PybindOfflineDolphinModelConfig(py::module *m) {
  using PyClass = OfflineDolphinModelConfig;
  py::class_<PyClass>(*m, "OfflineDolphinModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &>(),
           py::arg("model"), py::arg("encoder") = "", py::arg("decoder") = "",
           py::arg("language") = "", py::arg("region") = "")
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("language", &PyClass::language)
      .def_readwrite("region", &PyClass::region)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
