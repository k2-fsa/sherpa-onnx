// sherpa-onnx/python/csrc/offline-stream.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/offline-stream.h"

#include "sherpa-onnx/csrc/offline-stream.h"

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

static void PybindOfflineRecognitionResult(py::module *m) {  // NOLINT
  using PyClass = OfflineRecognitionResult;
  py::class_<PyClass>(*m, "OfflineRecognitionResult")
      .def_property_readonly(
          "text",
          [](const PyClass &self) -> py::str {
            return py::str(PyUnicode_DecodeUTF8(self.text.c_str(),
                                                self.text.size(), "ignore"));
          })
      .def_property_readonly("tokens",
                             [](const PyClass &self) { return self.tokens; })
      .def_property_readonly(
          "timestamps", [](const PyClass &self) { return self.timestamps; });
}

static void PybindOfflineFeatureExtractorConfig(py::module *m) {
  using PyClass = OfflineFeatureExtractorConfig;
  py::class_<PyClass>(*m, "OfflineFeatureExtractorConfig")
      .def(py::init<int32_t, int32_t>(), py::arg("sampling_rate") = 16000,
           py::arg("feature_dim") = 80)
      .def_readwrite("sampling_rate", &PyClass::sampling_rate)
      .def_readwrite("feature_dim", &PyClass::feature_dim)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineStream(py::module *m) {
  PybindOfflineFeatureExtractorConfig(m);
  PybindOfflineRecognitionResult(m);

  using PyClass = OfflineStream;
  py::class_<PyClass>(*m, "OfflineStream")
      .def(
          "accept_waveform",
          [](PyClass &self, float sample_rate, py::array_t<float> waveform) {
#if 0
            auto report_gil_status = []() {
              auto is_gil_held = false;
              if (auto tstate = py::detail::get_thread_state_unchecked())
                is_gil_held = (tstate == PyGILState_GetThisThreadState());

              return is_gil_held ? "GIL held" : "GIL released";
            };
            std::cout << report_gil_status() << "\n";
#endif
            self.AcceptWaveform(sample_rate, waveform.data(), waveform.size());
          },
          py::arg("sample_rate"), py::arg("waveform"), kAcceptWaveformUsage,
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("result", &PyClass::GetResult);
}

}  // namespace sherpa_onnx
