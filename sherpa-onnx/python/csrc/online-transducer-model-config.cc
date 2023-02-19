// sherpa-onnx/python/csrc/online-transducer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-transducer-model-config.h"

#include <string>

#include "sherpa-onnx/python/csrc/online-transducer-model-config.h"

namespace sherpa_onnx {

void PybindOnlineTransducerModelConfig(py::module *m) {
  using PyClass = OnlineTransducerModelConfig;
  py::class_<PyClass>(*m, "OnlineTransducerModelConfig")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, int32_t, bool>(),
           py::arg("encoder_filename"), py::arg("decoder_filename"),
           py::arg("joiner_filename"), py::arg("num_threads"),
           py::arg("debug") = false)
      .def_readwrite("encoder_filename", &PyClass::encoder_filename)
      .def_readwrite("decoder_filename", &PyClass::decoder_filename)
      .def_readwrite("joiner_filename", &PyClass::joiner_filename)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
