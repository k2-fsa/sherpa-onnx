// sherpa-onnx/python/csrc/offline-speech-denoiser.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-speech-denoiser.h"

#include <vector>

#include "sherpa-onnx/csrc/offline-speech-denoiser.h"
#include "sherpa-onnx/python/csrc/offline-speech-denoiser-model-config.h"

namespace sherpa_onnx {

static constexpr const char *kDenoisedAudioDoc = R"doc(
Represents denoised audio output.

Attributes:
  sample_rate:
    The sample rate of the denoised audio.
  samples:
    A 1-D float32 array of denoised audio samples.
)doc";

static constexpr const char *kOfflineSpeechDenoiserInitDoc = R"doc(
Constructor for OfflineSpeechDenoiser.

Args:
  config:
    Config for offline speech denoiser.
)doc";

static constexpr const char *kOfflineSpeechDenoiserRunDoc = R"doc(
Denoise the given audio samples.

Args:
  samples:
    A 1-D float32 array of audio samples.
  sample_rate:
    The sample rate of the input audio.

Returns:
  A DenoisedAudio object containing the denoised audio.
)doc";

void PybindOfflineSpeechDenoiserConfig(py::module *m) {
  PybindOfflineSpeechDenoiserModelConfig(m);

  using PyClass = OfflineSpeechDenoiserConfig;

  py::class_<PyClass>(*m, "OfflineSpeechDenoiserConfig")
      .def(py::init<>())
      .def(py::init<const OfflineSpeechDenoiserModelConfig &>(),
           py::arg("model") = OfflineSpeechDenoiserModelConfig{})
      .def_readwrite("model", &PyClass::model)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindDenoisedAudio(py::module *m) {
  using PyClass = DenoisedAudio;
  py::class_<PyClass>(*m, "DenoisedAudio", kDenoisedAudioDoc)
      .def_property_readonly(
          "sample_rate", [](const PyClass &self) { return self.sample_rate; })
      .def_property_readonly("samples",
                             [](const PyClass &self) { return self.samples; });
}

void PybindOfflineSpeechDenoiser(py::module *m) {
  PybindOfflineSpeechDenoiserConfig(m);
  PybindDenoisedAudio(m);
  using PyClass = OfflineSpeechDenoiser;
  py::class_<PyClass>(*m, "OfflineSpeechDenoiser")
      .def(py::init<const OfflineSpeechDenoiserConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>(),
           kOfflineSpeechDenoiserInitDoc)
      .def(
          "__call__",
          [](const PyClass &self, const std::vector<float> &samples,
             int32_t sample_rate) {
            return self.Run(samples.data(), samples.size(), sample_rate);
          },
          py::arg("samples"), py::arg("sample_rate"),
          py::call_guard<py::gil_scoped_release>(), kOfflineSpeechDenoiserRunDoc)
      .def(
          "run",
          [](const PyClass &self, const std::vector<float> &samples,
             int32_t sample_rate) {
            return self.Run(samples.data(), samples.size(), sample_rate);
          },
          py::arg("samples"), py::arg("sample_rate"),
          py::call_guard<py::gil_scoped_release>(), kOfflineSpeechDenoiserRunDoc)
      .def_property_readonly("sample_rate", &PyClass::GetSampleRate);
}

}  // namespace sherpa_onnx
