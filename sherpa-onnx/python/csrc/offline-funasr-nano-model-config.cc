// sherpa-onnx/python/csrc/offline-funasr-nano-model-config.cc
//
// Copyright (c)  2025  zengyw

#include "sherpa-onnx/csrc/offline-funasr-nano-model-config.h"

#include <string>

#include "sherpa-onnx/python/csrc/offline-funasr-nano-model-config.h"

namespace sherpa_onnx {

void PybindOfflineFunASRNanoModelConfig(py::module *m) {
  using PyClass = OfflineFunASRNanoModelConfig;
  py::class_<PyClass>(*m, "OfflineFunASRNanoModelConfig")
      .def(py::init<>())
      .def_readwrite("encoder_adaptor", &PyClass::encoder_adaptor)
      .def_readwrite("llm_prefill", &PyClass::llm_prefill)
      .def_readwrite("llm_decode", &PyClass::llm_decode)
      .def_readwrite("embedding", &PyClass::embedding)
      .def_readwrite("tokenizer", &PyClass::tokenizer)
      .def_readwrite("system_prompt", &PyClass::system_prompt)
      .def_readwrite("user_prompt", &PyClass::user_prompt)
      .def_readwrite("max_new_tokens", &PyClass::max_new_tokens)
      .def_readwrite("temperature", &PyClass::temperature)
      .def_readwrite("top_p", &PyClass::top_p)
      .def_readwrite("seed", &PyClass::seed)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx

