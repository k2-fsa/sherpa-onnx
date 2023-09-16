// sherpa-onnx/python/csrc/circular-buffer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/circular-buffer.h"

#include <vector>

#include "sherpa-onnx/csrc/circular-buffer.h"

namespace sherpa_onnx {

void PybindCircularBuffer(py::module *m) {
  using PyClass = CircularBuffer;
  py::class_<PyClass>(*m, "CircularBuffer")
      .def(py::init<int32_t>(), py::arg("capacity"))
      .def(
          "push",
          [](PyClass &self, const std::vector<float> &samples) {
            self.Push(samples.data(), samples.size());
          },
          py::arg("samples"))
      .def("get", &PyClass::Get, py::arg("start_index"), py::arg("n"))
      .def("pop", &PyClass::Pop, py::arg("n"))
      .def("reset", &PyClass::Reset)
      .def_property_readonly("size", &PyClass::Size)
      .def_property_readonly("head", &PyClass::Head)
      .def_property_readonly("tail", &PyClass::Tail);
}

}  // namespace sherpa_onnx
