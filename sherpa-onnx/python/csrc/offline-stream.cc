// sherpa-onnx/python/csrc/offline-stream.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/offline-stream.h"

#include <vector>

#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {

constexpr const char *kAcceptWaveformUsage = R"doc(
Process audio samples.

Args:
  sample_rate:
    Sample rate of the input samples. If it is different from the one
    expected by the model, we will do resampling inside.
  waveform:
    A 1-D float32 tensor containing audio samples. It must be normalized
    to the range [-1, 1].
)doc";

static constexpr const char *kOfflineRecognitionResultDoc = R"doc(
The result of offline (non-streaming) speech recognition.

Use the read-only properties to access individual fields of the result
such as text, tokens, timestamps, etc.
)doc";

static constexpr const char *kOfflineStreamDoc = R"doc(
An offline stream is created by an OfflineRecognizer and holds a
single utterance of audio. Feed audio into it with ``accept_waveform``,
then decode the stream with the recognizer to obtain the result.
)doc";

static constexpr const char *kSetOptionDoc = R"doc(
Set a per-stream option.

Args:
  key:
    The option name.
  value:
    The option value.
)doc";

static constexpr const char *kHasOptionDoc = R"doc(
Check whether a per-stream option is set.

Args:
  key:
    The option name.

Return:
  True if the option exists, False otherwise.
)doc";

static constexpr const char *kGetOptionDoc = R"doc(
Get the value of a per-stream option.

Args:
  key:
    The option name.

Return:
  The option value as a string, or an empty string if the key
  does not exist.
)doc";

static void PybindOfflineRecognitionResult(py::module *m) {  // NOLINT
  using PyClass = OfflineRecognitionResult;
  py::class_<PyClass>(*m, "OfflineRecognitionResult",
                      kOfflineRecognitionResultDoc)
      .def("__str__", &PyClass::AsJsonString)
      .def_property_readonly(
          "text",
          [](const PyClass &self) -> py::str {
            return py::str(PyUnicode_DecodeUTF8(self.text.c_str(),
                                                self.text.size(), "ignore"));
          })
      .def_property_readonly("lang",
         [](const PyClass &self) { return self.lang; })
      .def_property_readonly("emotion",
        [](const PyClass &self) { return self.emotion; })
      .def_property_readonly("event",
        [](const PyClass &self) { return self.event; })
      .def_property_readonly("tokens",
        [](const PyClass &self) { return self.tokens; })
      .def_property_readonly("words",
        [](const PyClass &self) { return self.words; })
      .def_property_readonly("timestamps",
        [](const PyClass &self) { return self.timestamps; })
      .def_property_readonly("durations",
        [](const PyClass &self) { return self.durations; })
      .def_property_readonly("ys_log_probs",
        [](const PyClass &self) { return self.ys_log_probs; })
      .def_property_readonly("segment_timestamps",
        [](const PyClass &self) { return self.segment_timestamps; })
      .def_property_readonly("segment_durations",
        [](const PyClass &self) { return self.segment_durations; })
      .def_property_readonly("segment_texts",
        [](const PyClass &self) { return self.segment_texts; });
}

void PybindOfflineStream(py::module *m) {
  PybindOfflineRecognitionResult(m);

  using PyClass = OfflineStream;
  py::class_<PyClass>(*m, "OfflineStream", kOfflineStreamDoc)
      .def(
          "accept_waveform",
          [](PyClass &self, float sample_rate,
             const std::vector<float> &waveform) {
            self.AcceptWaveform(sample_rate, waveform.data(), waveform.size());
          },
          py::arg("sample_rate"), py::arg("waveform"), kAcceptWaveformUsage,
          py::call_guard<py::gil_scoped_release>())
      .def("set_option", &PyClass::SetOption, py::arg("key"),
           py::arg("value"), kSetOptionDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("has_option", &PyClass::HasOption, py::arg("key"),
           kHasOptionDoc, py::call_guard<py::gil_scoped_release>())
      .def("get_option", &PyClass::GetOption, py::arg("key"),
           kGetOptionDoc, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("result", &PyClass::GetResult);
}

}  // namespace sherpa_onnx
