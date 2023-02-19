// sherpa-onnx/python/csrc/online-stream.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/online-stream.h"

#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

void PybindOnlineStream(py::module *m) {
  using PyClass = OnlineStream;
  py::class_<PyClass>(*m, "OnlineStream")
      .def("accept_waveform",
           [](PyClass &self, float sample_rate, py::array_t<float> waveform) {
             self.AcceptWaveform(sample_rate, waveform.data(), waveform.size());
           })
      .def("input_finished", &PyClass::InputFinished);
}

}  // namespace sherpa_onnx
