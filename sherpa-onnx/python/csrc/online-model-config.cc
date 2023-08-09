// sherpa-onnx/python/csrc/online-model-config.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/online-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model-config.h"
#include "sherpa-onnx/python/csrc/online-transducer-model-config.h"

namespace sherpa_onnx {

void PybindOnlineModelConfig(py::module *m) {
  PybindOnlineTransducerModelConfig(m);

  using PyClass = OnlineModelConfig;
  py::class_<PyClass>(*m, "OnlineModelConfig")
      .def(py::init<const OnlineTransducerModelConfig &, std::string &, int32_t,
                    bool, const std::string &, const std::string &>(),
           py::arg("transducer") = OnlineTransducerModelConfig(),
           py::arg("tokens"), py::arg("num_threads"), py::arg("debug") = false,
           py::arg("provider") = "cpu", py::arg("model_type") = "")
      .def_readwrite("transducer", &PyClass::transducer)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def_readwrite("model_type", &PyClass::model_type)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
