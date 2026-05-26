// sherpa-onnx/python/csrc/online-speech-denoiser.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/online-speech-denoiser.h"

#include <vector>

#include "sherpa-onnx/csrc/online-speech-denoiser.h"
#include "sherpa-onnx/python/csrc/offline-speech-denoiser.h"
#include "sherpa-onnx/python/csrc/offline-speech-denoiser-model-config.h"

namespace sherpa_onnx {

static void PybindOnlineSpeechDenoiserConfig(py::module *m) {
  using PyClass = OnlineSpeechDenoiserConfig;

  py::class_<PyClass>(*m, "OnlineSpeechDenoiserConfig")
      .def(py::init<>())
      .def(py::init<const OfflineSpeechDenoiserModelConfig &>(),
           py::arg("model") = OfflineSpeechDenoiserModelConfig{})
      .def_readwrite("model", &PyClass::model)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindOnlineSpeechDenoiser(py::module *m) {
  PybindOnlineSpeechDenoiserConfig(m);

  using PyClass = OnlineSpeechDenoiser;
  py::class_<PyClass>(*m, "OnlineSpeechDenoiser")
      .def(py::init<const OnlineSpeechDenoiserConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "__call__",
          [](PyClass &self, const std::vector<float> &samples,
             int32_t sample_rate) {
            return self.Run(samples.data(), samples.size(), sample_rate);
          },
          py::arg("samples"), py::arg("sample_rate"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "run",
          [](PyClass &self, const std::vector<float> &samples,
             int32_t sample_rate) {
            return self.Run(samples.data(), samples.size(), sample_rate);
          },
          py::arg("samples"), py::arg("sample_rate"),
          py::call_guard<py::gil_scoped_release>())
      .def("flush", &PyClass::Flush, py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("sample_rate", &PyClass::GetSampleRate)
      .def_property_readonly("frame_shift_in_samples",
                             &PyClass::GetFrameShiftInSamples);
}

}  // namespace sherpa_onnx
