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

static constexpr const char *kAcceptWaveformDoc = R"doc(
Feed audio samples to the VAD.

Args:
  samples:
    A 1-D float32 array of audio samples. The sample rate must match
    the one configured in ``VadModelConfig``.
)doc";

static constexpr const char *kIsSpeechDetectedDoc = R"doc(
Return True if speech is currently being detected.
)doc";

static constexpr const char *kEmptyDoc = R"doc(
Return True if there are no queued speech segments.
)doc";

static constexpr const char *kFrontDoc = R"doc(
Return the first queued speech segment.

It is an error to access this property when ``empty()`` returns True.
)doc";

static constexpr const char *kPopDoc = R"doc(
Remove the first queued speech segment.

It is an error to call this when ``empty()`` returns True.
)doc";

static constexpr const char *kFlushDoc = R"doc(
Flush the buffered tail samples so that the last segment is finalized.
)doc";

static constexpr const char *kResetDoc = R"doc(
Reset the VAD internal state. Call this before processing a new audio stream.
)doc";

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
          py::arg("samples"), kAcceptWaveformDoc,
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("config", &PyClass::GetConfig)
      .def("empty", &PyClass::Empty, kEmptyDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("pop", &PyClass::Pop, kPopDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("is_speech_detected", &PyClass::IsSpeechDetected,
           kIsSpeechDetectedDoc, py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, kResetDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("flush", &PyClass::Flush, kFlushDoc,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("front", &PyClass::Front, kFrontDoc)
      .def_property_readonly("current_segment", &PyClass::CurrentSpeechSegment);
}

}  // namespace sherpa_onnx
