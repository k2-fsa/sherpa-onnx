// sherpa-onnx/python/csrc/nomo-pvad-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/nomo-pvad-model-config.h"

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/nomo-pvad-model-config.h"

namespace sherpa_onnx {

void PybindNomoPvadModelConfig(py::module *m) {
  using PyClass = NomoPvadModelConfig;
  py::class_<PyClass>(*m, "NomoPvadModelConfig")
      .def(py::init<>())
      .def(py::init([](const std::string &model,
                       const std::vector<float> &speaker_embedding,
                       float threshold, float min_silence_duration,
                       float min_speech_duration, float max_speech_duration,
                       float neg_threshold) -> std::unique_ptr<PyClass> {
             auto ans = std::make_unique<PyClass>();

             ans->model = model;
             ans->speaker_embedding = speaker_embedding;
             ans->threshold = threshold;
             ans->min_silence_duration = min_silence_duration;
             ans->min_speech_duration = min_speech_duration;
             ans->max_speech_duration = max_speech_duration;
             ans->neg_threshold = neg_threshold;

             return ans;
           }),
           py::arg("model"), py::arg("speaker_embedding"),
           py::arg("threshold") = 0.7,
           py::arg("min_silence_duration") = 0.5,
           py::arg("min_speech_duration") = 0.25,
           py::arg("max_speech_duration") = 20,
           py::arg("neg_threshold") = -1)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("speaker_embedding", &PyClass::speaker_embedding)
      .def_readwrite("speaker_embedding_file", &PyClass::speaker_embedding_file)
      .def_readwrite("threshold", &PyClass::threshold)
      .def_readwrite("min_silence_duration", &PyClass::min_silence_duration)
      .def_readwrite("min_speech_duration", &PyClass::min_speech_duration)
      .def_readwrite("max_speech_duration", &PyClass::max_speech_duration)
      .def_readwrite("neg_threshold", &PyClass::neg_threshold)
      .def("__str__", &PyClass::ToString)
      .def("validate", static_cast<bool (NomoPvadModelConfig::*)() const>(
                           &PyClass::Validate));
}

}  // namespace sherpa_onnx
