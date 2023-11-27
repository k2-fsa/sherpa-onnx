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
      .def_static("create", &PyClass::Create, py::arg("config"),
                  py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, py::call_guard<py::gil_scoped_release>())
      .def(
          "is_speech",
          [](PyClass &self, const std::vector<float> &samples) -> bool {
            return self.IsSpeech(samples.data(), samples.size());
          },
          py::arg("samples"), py::call_guard<py::gil_scoped_release>())
      .def("window_size", &PyClass::WindowSize,
           py::call_guard<py::gil_scoped_release>())
      .def("min_silence_duration_samples", &PyClass::MinSilenceDurationSamples,
           py::call_guard<py::gil_scoped_release>())
      .def("min_speech_duration_samples", &PyClass::MinSpeechDurationSamples,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_onnx
