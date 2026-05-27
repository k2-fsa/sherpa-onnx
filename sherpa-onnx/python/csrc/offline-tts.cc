// sherpa-onnx/python/csrc/offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/python/csrc/offline-tts.h"

#include <algorithm>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/python/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

static void PybindGeneratedAudio(py::module *m) {
  using PyClass = GeneratedAudio;
  py::class_<PyClass>(*m, "GeneratedAudio")
      .def(py::init<>())
      .def_readwrite("samples", &PyClass::samples)
      .def_readwrite("sample_rate", &PyClass::sample_rate)
      .def("__str__", [](PyClass &self) {
        std::ostringstream os;
        os << "GeneratedAudio(sample_rate=" << self.sample_rate << ", ";
        os << "num_samples=" << self.samples.size() << ")";
        return os.str();
      });
}

static void PybindGenerationConfig(py::module *m) {
  using PyClass = GenerationConfig;

  py::class_<PyClass>(*m, "GenerationConfig")
      .def(py::init<>())
      .def_readwrite("silence_scale", &PyClass::silence_scale)
      .def_readwrite("speed", &PyClass::speed)
      .def_readwrite("sid", &PyClass::sid)
      .def_readwrite("reference_audio", &PyClass::reference_audio)
      .def_readwrite("reference_sample_rate", &PyClass::reference_sample_rate)
      .def_readwrite("reference_text", &PyClass::reference_text)
      .def_readwrite("num_steps", &PyClass::num_steps)
      .def_readwrite("extra", &PyClass::extra)
      .def("__str__", &PyClass::ToString);
}

static void PybindOfflineTtsConfig(py::module *m) {
  PybindOfflineTtsModelConfig(m);

  using PyClass = OfflineTtsConfig;
  py::class_<PyClass>(*m, "OfflineTtsConfig")
      .def(py::init<>())
      .def(py::init<const OfflineTtsModelConfig &, const std::string &,
                    const std::string &, int32_t, float>(),
           py::arg("model"), py::arg("rule_fsts") = "",
           py::arg("rule_fars") = "", py::arg("max_num_sentences") = 1,
           py::arg("silence_scale") = 0.2)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def_readwrite("rule_fars", &PyClass::rule_fars)
      .def_readwrite("max_num_sentences", &PyClass::max_num_sentences)
      .def_readwrite("silence_scale", &PyClass::silence_scale)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

static constexpr const char *kOfflineTtsDoc = R"doc(
Offline text-to-speech engine.

Args:
  config:
    The configuration for the offline TTS model.
)doc";

static constexpr const char *kGenerateDoc = R"doc(
Generate speech from text.

Args:
  text:
    The text to generate speech for.
  sid:
    Speaker ID. Used for multi-speaker models.
  speed:
    The speaking speed. Larger values produce faster speech.
  callback:
    If not None, it is called during speech generation with
    ``(samples: np.ndarray, progress: float) -> int``.
    Return a non-zero value to stop generation early.

Returns:
  A ``GeneratedAudio`` object containing the audio samples and sample rate.
)doc";

static constexpr const char *kSampleRateDoc = R"doc(
Return the sample rate of the generated audio.
)doc";

static constexpr const char *kNumSpeakersDoc = R"doc(
Return the number of speakers supported by the model.
)doc";

void PybindOfflineTts(py::module *m) {
  PybindOfflineTtsConfig(m);
  PybindGeneratedAudio(m);
  PybindGenerationConfig(m);

  using PyClass = OfflineTts;
  py::class_<PyClass>(*m, "OfflineTts", kOfflineTtsDoc)
      .def(py::init<const OfflineTtsConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("sample_rate", &PyClass::SampleRate,
                             kSampleRateDoc)
      .def_property_readonly("num_speakers", &PyClass::NumSpeakers,
                             kNumSpeakersDoc)
      .def(
          "generate",
          [](const PyClass &self, const std::string &text, int64_t sid,
             float speed,
             std::function<int32_t(py::array_t<float>, float)> callback)
              -> GeneratedAudio {
            if (!callback) {
              GenerationConfig config;
              config.sid = sid;
              config.speed = speed;
              return self.Generate(text, config);
            }

            std::function<int32_t(const float *, int32_t, float)>
                callback_wrapper = [callback](const float *samples, int32_t n,
                                              float progress) {
                  // CAUTION(fangjun): we have to copy samples since it is
                  // freed once the call back returns.

                  pybind11::gil_scoped_acquire acquire;

                  pybind11::array_t<float> array(n);
                  py::buffer_info buf = array.request();
                  auto p = static_cast<float *>(buf.ptr);
                  std::copy(samples, samples + n, p);
                  return callback(array, progress);
                };

            GenerationConfig config;
            config.sid = sid;
            config.speed = speed;
            return self.Generate(text, config, callback_wrapper);
          },
          py::arg("text"), py::arg("sid") = 0, py::arg("speed") = 1.0,
          py::arg("callback") = py::none(),
          kGenerateDoc, py::call_guard<py::gil_scoped_release>())
      .def(
          "generate",
          [](const PyClass &self, const std::string &text,
             const GenerationConfig &config,
             std::function<int32_t(py::array_t<float>, float)> callback)
              -> GeneratedAudio {
            if (!callback) {
              return self.Generate(text, config);
            }

            std::function<int32_t(const float *, int32_t, float)>
                callback_wrapper = [callback](const float *samples, int32_t n,
                                              float progress) {
                  py::gil_scoped_acquire acquire;

                  py::array_t<float> array(n);
                  auto buf = array.request();
                  auto *p = static_cast<float *>(buf.ptr);
                  std::copy(samples, samples + n, p);

                  return callback(array, progress);
                };

            return self.Generate(text, config, callback_wrapper);
          },
          py::arg("text"), py::arg("config"), py::arg("callback") = py::none(),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "generate",
          [](const PyClass &self, const std::string &text,
             const std::string &prompt_text,
             const std::vector<float> &prompt_samples, int32_t sample_rate,
             float speed, int32_t num_steps,
             std::function<int32_t(py::array_t<float>, float)> callback)
              -> GeneratedAudio {
            GenerationConfig config;
            config.reference_audio = prompt_samples;
            config.reference_sample_rate = sample_rate;
            config.reference_text = prompt_text;
            config.speed = speed;
            config.num_steps = num_steps;

            if (!callback) {
              return self.Generate(text, config);
            }

            std::function<int32_t(const float *, int32_t, float)>
                callback_wrapper = [callback](const float *samples, int32_t n,
                                              float progress) {
                  // CAUTION(fangjun): we have to copy samples since it is
                  // freed once the call back returns.

                  pybind11::gil_scoped_acquire acquire;

                  pybind11::array_t<float> array(n);
                  py::buffer_info buf = array.request();
                  auto p = static_cast<float *>(buf.ptr);
                  std::copy(samples, samples + n, p);
                  return callback(array, progress);
                };

            return self.Generate(text, config, callback_wrapper);
          },
          py::arg("text"), py::arg("prompt_text"), py::arg("prompt_samples"),
          py::arg("sample_rate"), py::arg("speed") = 1.0,
          py::arg("num_steps") = 4, py::arg("callback") = py::none(),
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_onnx
