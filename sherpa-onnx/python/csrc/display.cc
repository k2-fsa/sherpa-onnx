// sherpa-onnx/python/csrc/display.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/display.h"

#include "sherpa-onnx/csrc/display.h"

namespace sherpa_onnx {

static constexpr const char *kDisplayInitDoc = R"doc(
Constructor for Display.

Args:
  max_word_per_line:
    Maximum number of words per line for display.
)doc";

static constexpr const char *kDisplayPrintDoc = R"doc(
Display recognition results.

Args:
  idx:
    The segment index.
  s:
    The text to display.
)doc";

void PybindDisplay(py::module *m) {
  using PyClass = Display;
  py::class_<PyClass>(*m, "Display")
      .def(py::init<int32_t>(), py::arg("max_word_per_line") = 60,
           kDisplayInitDoc)
      .def("print", &PyClass::Print, py::arg("idx"), py::arg("s"),
           kDisplayPrintDoc);
}

}  // namespace sherpa_onnx
