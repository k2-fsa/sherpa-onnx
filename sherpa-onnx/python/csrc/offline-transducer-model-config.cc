// sherpa-onnx/python/csrc/offline-transducer-model-config.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/offline-transducer-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-transducer-model-config.h"

namespace sherpa_onnx {

void PybindOfflineTransducerModelConfig(py::module *m) {
  using PyClass = OfflineTransducerModelConfig;
  py::class_<PyClass>(*m, "OfflineTransducerModelConfig")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &>(),
           py::arg("encoder_filename"), py::arg("decoder_filename"),
           py::arg("joiner_filename"))
      .def_readwrite("encoder_filename", &PyClass::encoder_filename)
      .def_readwrite("decoder_filename", &PyClass::decoder_filename)
      .def_readwrite("joiner_filename", &PyClass::joiner_filename)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
