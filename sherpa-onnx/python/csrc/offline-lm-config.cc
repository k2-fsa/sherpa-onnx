// sherpa-onnx/python/csrc/offline-lm-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-lm-config.h"

#include <string>

#include "sherpa-onnx//csrc/offline-lm-config.h"

namespace sherpa_onnx {

void PybindOfflineLMConfig(py::module *m) {
  using PyClass = OfflineLMConfig;
  py::class_<PyClass>(*m, "OfflineLMConfig")
      .def(py::init<const std::string &, float, int32_t, const std::string &,
           const std::string &, float, int32_t>(),
           py::arg("model"), py::arg("scale") = 0.5f,
           py::arg("lm_num_threads") = 1, py::arg("lm_provider") = "cpu",
           py::arg("lodr_fst") = "", py::arg("lodr_scale") = 0.0f,
           py::arg("lodr_backoff_id") = -1)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("scale", &PyClass::scale)
      .def_readwrite("lm_provider", &PyClass::lm_provider)
      .def_readwrite("lm_num_threads", &PyClass::lm_num_threads)
      .def_readwrite("lodr_fst", &PyClass::lodr_fst)
      .def_readwrite("lodr_scale", &PyClass::lodr_scale)
      .def_readwrite("lodr_backoff_id", &PyClass::lodr_backoff_id)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
