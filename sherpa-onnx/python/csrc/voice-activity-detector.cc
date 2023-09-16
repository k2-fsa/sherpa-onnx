// sherpa-onnx/python/csrc/voice-activity-detector.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/voice-activity-detector.h"

#include <vector>

#include "sherpa-onnx/csrc/voice-activity-detector.h"

namespace sherpa_onnx {

void PybindSpeechSegment(py::module *m) {
  using PyClass = SpeechSegment;
  py::class_<PyClass>(*m, "SpeechSegment")
      .def_property_readonly("start",
                             [](const PyClass &self) { return self.start; })
      .def_property_readonly("samples",
                             [](const PyClass &self) { return self.samples; });
}

void PybindVoiceActivityDetector(py::module *m) {
  PybindSpeechSegment(m);
  using PyClass = VoiceActivityDetector;
  py::class_<PyClass>(*m, "VoiceActivityDetector")
      .def(py::init<const VadModelConfig &, float>(), py::arg("config"),
           py::arg("buffer_size_in_seconds") = 60)
      .def(
          "accept_waveform",
          [](PyClass &self, const std::vector<float> &samples) {
            self.AcceptWaveform(samples.data(), samples.size());
          },
          py::arg("samples"))
      .def("empty", &PyClass::Empty)
      .def("pop", &PyClass::Pop)
      .def("is_speech_detected", &PyClass::IsSpeechDetected)
      .def("reset", &PyClass::Reset)
      .def_property_readonly("front", &PyClass::Front);
}

}  // namespace sherpa_onnx
