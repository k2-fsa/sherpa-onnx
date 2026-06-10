// sherpa-onnx/python/csrc/offline-diacritization.cc
//
// Copyright (c)  2026  Matias Lin

#include "sherpa-onnx/python/csrc/offline-diacritization.h"

#include <string>

#include "sherpa-onnx/csrc/offline-diacritization.h"

namespace sherpa_onnx {

static constexpr const char *kOfflineDiacritizationInitDoc = R"doc(
Constructor for OfflineDiacritization.

Args:
  config:
    Config for offline diacritization.
)doc";

static constexpr const char *kOfflineDiacritizationAddDiacriticsDoc = R"doc(
Add diacritics to the given text.

Args:
  text:
    The input text without diacritics.

Returns:
  The text with diacritics added.
)doc";

static void PybindOfflineDiacritizationModelConfig(py::module *m) {
  using PyClass = OfflineDiacritizationModelConfig;
  py::class_<PyClass>(*m, "OfflineDiacritizationModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &, int32_t, bool,
                    const std::string &>(),
           py::arg("catt_encoder"), py::arg("catt_decoder"),
           py::arg("num_threads") = 1, py::arg("debug") = false,
           py::arg("provider") = "cpu")
      .def_readwrite("catt_encoder", &PyClass::catt_encoder)
      .def_readwrite("catt_decoder", &PyClass::catt_decoder)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

static void PybindOfflineDiacritizationConfig(py::module *m) {
  PybindOfflineDiacritizationModelConfig(m);
  using PyClass = OfflineDiacritizationConfig;

  py::class_<PyClass>(*m, "OfflineDiacritizationConfig")
      .def(py::init<>())
      .def(py::init<const OfflineDiacritizationModelConfig &>(),
           py::arg("model"))
      .def_readwrite("model", &PyClass::model)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineDiacritization(py::module *m) {
  PybindOfflineDiacritizationConfig(m);
  using PyClass = OfflineDiacritization;

  py::class_<PyClass>(*m, "OfflineDiacritization")
      .def(py::init<const OfflineDiacritizationConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>(),
           kOfflineDiacritizationInitDoc)
      .def("add_diacritics", &PyClass::AddDiacritics, py::arg("text"),
           py::call_guard<py::gil_scoped_release>(),
           kOfflineDiacritizationAddDiacriticsDoc);
}

}  // namespace sherpa_onnx
