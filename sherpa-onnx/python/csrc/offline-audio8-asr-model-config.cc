// sherpa-onnx/python/csrc/offline-audio8-asr-model-config.cc
//
// Copyright (c)  2026  yuyun2000

#include "sherpa-onnx/python/csrc/offline-audio8-asr-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-audio8-asr-model-config.h"

namespace sherpa_onnx {

void PybindOfflineAudio8AsrModelConfig(py::module *m) {
  using PyClass = OfflineAudio8AsrModelConfig;
  py::class_<PyClass>(*m, "OfflineAudio8AsrModelConfig")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, int32_t>(),
           py::arg("encoder") = "", py::arg("decoder") = "",
           py::arg("tokenizer") = "", py::arg("max_new_tokens") = 128)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("tokenizer", &PyClass::tokenizer)
      .def_readwrite("max_new_tokens", &PyClass::max_new_tokens)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
