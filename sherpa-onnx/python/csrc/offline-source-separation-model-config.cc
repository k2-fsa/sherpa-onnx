// sherpa-onnx/python/csrc/offline-source-separation-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-source-separation-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-source-separation-model-config.h"
#include "sherpa-onnx/python/csrc/offline-source-separation-spleeter-model-config.h"
#include "sherpa-onnx/python/csrc/offline-source-separation-uvr-model-config.h"

namespace sherpa_onnx {

void PybindOfflineSourceSeparationModelConfig(py::module *m) {
  PybindOfflineSourceSeparationSpleeterModelConfig(m);
  PybindOfflineSourceSeparationUvrModelConfig(m);

  using PyClass = OfflineSourceSeparationModelConfig;
  py::class_<PyClass>(*m, "OfflineSourceSeparationModelConfig")
      .def(py::init<const OfflineSourceSeparationSpleeterModelConfig &,
                    const OfflineSourceSeparationUvrModelConfig &, int32_t,
                    bool, const std::string &>(),
           py::arg("spleeter") = OfflineSourceSeparationSpleeterModelConfig{},
           py::arg("uvr") = OfflineSourceSeparationUvrModelConfig{},
           py::arg("num_threads") = 1, py::arg("debug") = false,
           py::arg("provider") = "cpu")
      .def_readwrite("spleeter", &PyClass::spleeter)
      .def_readwrite("uvr", &PyClass::uvr)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
