// sherpa-onnx/python/csrc/offline-speech-denoiser-dpdfnet-model-config.cc
//
// Copyright (c)  2026  Ceva Inc

#include "sherpa-onnx/python/csrc/offline-speech-denoiser-dpdfnet-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model-config.h"

namespace sherpa_onnx {

void PybindOfflineSpeechDenoiserDpdfNetModelConfig(py::module *m) {
  using PyClass = OfflineSpeechDenoiserDpdfNetModelConfig;
  py::class_<PyClass>(*m, "OfflineSpeechDenoiserDpdfNetModelConfig")
      .def(py::init<>())
      .def(py::init([](const std::string &model, float attenuation_limit_db) {
             PyClass ans;
             ans.model = model;
             ans.attenuation_limit_db = attenuation_limit_db;
             return ans;
           }),
           py::arg("model") = "", py::arg("attenuation_limit_db") = 0.0f)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("attenuation_limit_db", &PyClass::attenuation_limit_db)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
