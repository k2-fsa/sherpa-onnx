// sherpa-onnx/python/csrc/spoken-language-identification.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/spoken-language-identification.h"

#include <string>

#include "sherpa-onnx/csrc/spoken-language-identification.h"

namespace sherpa_onnx {

static constexpr const char *kSpokenLanguageIdentificationInitDoc = R"doc(
Constructor for SpokenLanguageIdentification.

Args:
  config:
    Config for spoken language identification.
)doc";

static constexpr const char *kSpokenLanguageIdentificationCreateStreamDoc = R"doc(
Create a stream for feeding audio data.

Returns:
  An OnlineStream object.
)doc";

static constexpr const char *kSpokenLanguageIdentificationComputeDoc = R"doc(
Identify the language of the audio in the given stream.

Args:
  s:
    The stream containing audio data.

Returns:
  A string representing the identified language code.
)doc";

static void PybindSpokenLanguageIdentificationWhisperConfig(py::module *m) {
  using PyClass = SpokenLanguageIdentificationWhisperConfig;

  py::class_<PyClass>(*m, "SpokenLanguageIdentificationWhisperConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &, int32_t>(),
           py::arg("encoder"), py::arg("decoder"),
           py::arg("tail_paddings") = -1)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("decoder", &PyClass::decoder)
      .def_readwrite("tail_paddings", &PyClass::tail_paddings)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

static void PybindSpokenLanguageIdentificationConfig(py::module *m) {
  PybindSpokenLanguageIdentificationWhisperConfig(m);

  using PyClass = SpokenLanguageIdentificationConfig;

  py::class_<PyClass>(*m, "SpokenLanguageIdentificationConfig")
      .def(py::init<>())
      .def(py::init<const SpokenLanguageIdentificationWhisperConfig &, int32_t,
                    bool, const std::string &>(),
           py::arg("whisper"), py::arg("num_threads") = 1,
           py::arg("debug") = false, py::arg("provider") = "cpu")
      .def_readwrite("whisper", &PyClass::whisper)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindSpokenLanguageIdentification(py::module *m) {
  PybindSpokenLanguageIdentificationConfig(m);

  using PyClass = SpokenLanguageIdentification;
  py::class_<PyClass>(*m, "SpokenLanguageIdentification")
      .def(py::init<const SpokenLanguageIdentificationConfig &>(),
           py::arg("config"), py::call_guard<py::gil_scoped_release>(),
           kSpokenLanguageIdentificationInitDoc)
      .def("create_stream", &PyClass::CreateStream,
           py::call_guard<py::gil_scoped_release>(),
           kSpokenLanguageIdentificationCreateStreamDoc)
      .def("compute", &PyClass::Compute, py::arg("s"),
           py::call_guard<py::gil_scoped_release>(),
           kSpokenLanguageIdentificationComputeDoc);
}

}  // namespace sherpa_onnx
