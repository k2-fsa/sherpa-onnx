// sherpa-onnx/python/csrc/offline-model-config.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/offline-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/python/csrc/offline-transducer-model-config.h"
#include "sherpa-onnx/python/csrc/offline-paraformer-model-config.h"

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

void PybindOfflineModelConfig(py::module *m) {
  PybindOfflineTransducerModelConfig(m);
  PybindOfflineParaformerModelConfig(m);

  using PyClass = OfflineModelConfig;
  py::class_<PyClass>(*m, "OfflineModelConfig")
      .def(py::init<OfflineTransducerModelConfig &,
              OfflineParaformerModelConfig &,
              const std::string &, int32_t, bool>(),
           py::arg("transducer"), py::arg("paraformer"), py::arg("tokens"),
           py::arg("num_threads"), py::arg("debug") = false)
      .def_readwrite("transducer", &PyClass::transducer)
      .def_readwrite("paraformer", &PyClass::paraformer)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
