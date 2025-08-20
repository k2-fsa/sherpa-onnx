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
  py::class_<PyClass>(*m, "VoiceActivityDetector",
                      R"(
1. It is an error to call the front property when the method empty() returns True
2. The property front returns a reference, which is valid until the next call of any
   methods of this class
3. When speech is detected, the method is_speech_detected() return True, you can
   use the property current_segment to get the speech samples since
   is_speech_detected() returns true
4. When is_speech_detected() is changed from True to False, the method
   empty() returns False.
      )")
      .def(py::init<const VadModelConfig &, float>(), py::arg("config"),
           py::arg("buffer_size_in_seconds") = 60,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "accept_waveform",
          [](PyClass &self, const std::vector<float> &samples) {
            self.AcceptWaveform(samples.data(), samples.size());
          },
          py::arg("samples"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("config", &PyClass::GetConfig)
      .def("empty", &PyClass::Empty, py::call_guard<py::gil_scoped_release>())
      .def("pop", &PyClass::Pop, py::call_guard<py::gil_scoped_release>())
      .def("is_speech_detected", &PyClass::IsSpeechDetected,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, py::call_guard<py::gil_scoped_release>())
      .def("flush", &PyClass::Flush, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("front", &PyClass::Front)
      .def_property_readonly("current_segment", &PyClass::CurrentSpeechSegment);
}

}  // namespace sherpa_onnx
