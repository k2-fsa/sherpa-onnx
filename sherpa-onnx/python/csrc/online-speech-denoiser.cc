// sherpa-onnx/python/csrc/online-speech-denoiser.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/online-speech-denoiser.h"

#include <vector>

#include "sherpa-onnx/csrc/online-speech-denoiser.h"
#include "sherpa-onnx/python/csrc/offline-speech-denoiser.h"
#include "sherpa-onnx/python/csrc/offline-speech-denoiser-model-config.h"

namespace sherpa_onnx {

static constexpr const char *kOnlineSpeechDenoiserInitDoc = R"doc(
Constructor for OnlineSpeechDenoiser.

Args:
  config:
    Config for online speech denoiser.
)doc";

static constexpr const char *kOnlineSpeechDenoiserRunDoc = R"doc(
Feed audio samples to the denoiser and return denoised audio.

Args:
  samples:
    A 1-D float32 array of audio samples.
  sample_rate:
    The sample rate of the input audio.

Returns:
  A DenoisedAudio object containing the denoised audio.
)doc";

static constexpr const char *kOnlineSpeechDenoiserFlushDoc = R"doc(
Flush remaining audio data in the buffer and return denoised audio.

Returns:
  A DenoisedAudio object containing the denoised audio.
)doc";

static constexpr const char *kOnlineSpeechDenoiserResetDoc = R"doc(
Reset the state of the online speech denoiser.
)doc";

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
           py::call_guard<py::gil_scoped_release>(),
           kOnlineSpeechDenoiserInitDoc)
      .def(
          "__call__",
          [](PyClass &self, const std::vector<float> &samples,
             int32_t sample_rate) {
            return self.Run(samples.data(), samples.size(), sample_rate);
          },
          py::arg("samples"), py::arg("sample_rate"),
          py::call_guard<py::gil_scoped_release>(), kOnlineSpeechDenoiserRunDoc)
      .def(
          "run",
          [](PyClass &self, const std::vector<float> &samples,
             int32_t sample_rate) {
            return self.Run(samples.data(), samples.size(), sample_rate);
          },
          py::arg("samples"), py::arg("sample_rate"),
          py::call_guard<py::gil_scoped_release>(), kOnlineSpeechDenoiserRunDoc)
      .def("flush", &PyClass::Flush, py::call_guard<py::gil_scoped_release>(),
           kOnlineSpeechDenoiserFlushDoc)
      .def("reset", &PyClass::Reset, py::call_guard<py::gil_scoped_release>(),
           kOnlineSpeechDenoiserResetDoc)
      .def_property_readonly("sample_rate", &PyClass::GetSampleRate)
      .def_property_readonly("frame_shift_in_samples",
                             &PyClass::GetFrameShiftInSamples);
}

}  // namespace sherpa_onnx
