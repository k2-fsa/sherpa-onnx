// sherpa-onnx/python/csrc/alsa.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/alsa.h"

#include <vector>

#include "sherpa-onnx/csrc/alsa.h"

namespace sherpa_onnx {

void PybindAlsa(py::module *m) {
  using PyClass = Alsa;
  py::class_<PyClass>(*m, "Alsa")
      .def(py::init<const char *>(), py::arg("device_name"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "read",
          [](PyClass &self, int32_t num_samples) -> std::vector<float> {
            return self.Read(num_samples);
          },
          py::arg("num_samples"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("expected_sample_rate",
                             &PyClass::GetExpectedSampleRate)
      .def_property_readonly("actual_sample_rate",
                             &PyClass::GetActualSampleRate);
}

}  // namespace sherpa_onnx
