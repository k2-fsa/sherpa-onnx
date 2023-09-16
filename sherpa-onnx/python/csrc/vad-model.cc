// sherpa-onnx/python/csrc/vad-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/vad-model.h"

#include <vector>

#include "sherpa-onnx/csrc/vad-model.h"

namespace sherpa_onnx {

void PybindVadModel(py::module *m) {
  using PyClass = VadModel;
  py::class_<PyClass>(*m, "VadModel")
      .def_static("create", &PyClass::Create, py::arg("config"))
      .def("reset", &PyClass::Reset)
      .def(
          "is_speech",
          [](PyClass &self, const std::vector<float> &samples) -> bool {
            return self.IsSpeech(samples.data(), samples.size());
          },
          py::arg("samples"))
      .def("window_size", &PyClass::WindowSize)
      .def("min_silence_duration_samples", &PyClass::MinSilenceDurationSamples)
      .def("min_speech_duration_samples", &PyClass::MinSpeechDurationSamples);
}

}  // namespace sherpa_onnx
