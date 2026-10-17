// sherpa-onnx/python/csrc/offline-qwen3-asr-model-config.cc
//
// Copyright (c)  2026  zengyw

#include "sherpa-onnx/csrc/offline-qwen3-asr-model-config.h"

#include <string>

#include "sherpa-onnx/python/csrc/offline-qwen3-asr-model-config.h"

namespace sherpa_onnx {

void PybindOfflineQwen3ASRModelConfig(py::module *m) {
  using PyClass = OfflineQwen3ASRModelConfig;
  py::class_<PyClass>(*m, "OfflineQwen3ASRModelConfig")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &, int32_t, int32_t,
                    float, float, int32_t, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &>(),
           py::arg("conv_frontend") = "", py::arg("encoder") = "",
           py::arg("decoder") = "", py::arg("tokenizer") = "",
           py::arg("max_total_len") = 512, py::arg("max_new_tokens") = 128,
           py::arg("temperature") = 1e-6f, py::arg("top_p") = 0.8f,
           py::arg("seed") = 42, py::arg("hotwords") = "",
           py::arg("forced_aligner_conv_frontend") = "",
           py::arg("forced_aligner_encoder") = "",
           py::arg("forced_aligner_decoder") = "",
           py::arg("forced_aligner_tokenizer") = "")
      .def_readwrite("conv_frontend", &PyClass::conv_frontend)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("tokenizer", &PyClass::tokenizer)
      .def_readwrite("hotwords", &PyClass::hotwords)
      .def_readwrite("forced_aligner_conv_frontend",
                     &PyClass::forced_aligner_conv_frontend)
      .def_readwrite("forced_aligner_encoder", &PyClass::forced_aligner_encoder)
      .def_readwrite("forced_aligner_decoder", &PyClass::forced_aligner_decoder)
      .def_readwrite("forced_aligner_tokenizer",
                     &PyClass::forced_aligner_tokenizer)
      .def_readwrite("max_total_len", &PyClass::max_total_len)
      .def_readwrite("max_new_tokens", &PyClass::max_new_tokens)
      .def_readwrite("temperature", &PyClass::temperature)
      .def_readwrite("top_p", &PyClass::top_p)
      .def_readwrite("seed", &PyClass::seed)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
