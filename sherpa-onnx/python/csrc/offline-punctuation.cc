// sherpa-onnx/python/csrc/offline-punctuation.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-punctuation.h"

#include <string>

#include "sherpa-onnx/csrc/offline-punctuation.h"

namespace sherpa_onnx {

static constexpr const char *kOfflinePunctuationInitDoc = R"doc(
Constructor for OfflinePunctuation.

Args:
  config:
    Config for offline punctuation.
)doc";

static constexpr const char *kOfflinePunctuationAddPunctuationDoc = R"doc(
Add punctuation to the given text.

Args:
  text:
    The input text without punctuation.

Returns:
  The text with punctuation added.
)doc";

static void PybindOfflinePunctuationModelConfig(py::module *m) {
  using PyClass = OfflinePunctuationModelConfig;
  py::class_<PyClass>(*m, "OfflinePunctuationModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, int32_t, bool, const std::string &>(),
           py::arg("ct_transformer"), py::arg("num_threads") = 1,
           py::arg("debug") = false, py::arg("provider") = "cpu")
      .def_readwrite("ct_transformer", &PyClass::ct_transformer)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

static void PybindOfflinePunctuationConfig(py::module *m) {
  PybindOfflinePunctuationModelConfig(m);
  using PyClass = OfflinePunctuationConfig;

  py::class_<PyClass>(*m, "OfflinePunctuationConfig")
      .def(py::init<>())
      .def(py::init<const OfflinePunctuationModelConfig &>(), py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflinePunctuation(py::module *m) {
  PybindOfflinePunctuationConfig(m);
  using PyClass = OfflinePunctuation;

  py::class_<PyClass>(*m, "OfflinePunctuation")
      .def(py::init<const OfflinePunctuationConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>(),
           kOfflinePunctuationInitDoc)
      .def("add_punctuation", &PyClass::AddPunctuation, py::arg("text"),
           py::call_guard<py::gil_scoped_release>(),
           kOfflinePunctuationAddPunctuationDoc);
}

}  // namespace sherpa_onnx
