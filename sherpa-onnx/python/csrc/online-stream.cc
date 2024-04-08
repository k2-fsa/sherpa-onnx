// sherpa-onnx/python/csrc/online-stream.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/online-stream.h"

#include <vector>

#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

constexpr const char *kAcceptWaveformUsage = R"(
Process audio samples.

Args:
  sample_rate:
    Sample rate of the input samples. If it is different from the one
    expected by the model, we will do resampling inside.
  waveform:
    A 1-D float32 tensor containing audio samples. It must be normalized
    to the range [-1, 1].
)";

void PybindOnlineStream(py::module *m) {
  using PyClass = OnlineStream;
  py::class_<PyClass>(*m, "OnlineStream")
      .def(
          "accept_waveform",
          [](PyClass &self, float sample_rate,
             const std::vector<float> &waveform) {
            self.AcceptWaveform(sample_rate, waveform.data(), waveform.size());
          },
          py::arg("sample_rate"), py::arg("waveform"), kAcceptWaveformUsage,
          py::call_guard<py::gil_scoped_release>())
      .def("input_finished", &PyClass::InputFinished,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_onnx
