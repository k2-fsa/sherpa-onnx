// sherpa-onnx/python/csrc/offline-recognizer.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/offline-recognizer.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-recognizer.h"

namespace sherpa_onnx {

static void PybindOfflineRecognizerConfig(py::module *m) {
  using PyClass = OfflineRecognizerConfig;
  py::class_<PyClass>(*m, "OfflineRecognizerConfig")
      .def(py::init<const FeatureExtractorConfig &, const OfflineModelConfig &,
                    const OfflineLMConfig &, const OfflineCtcFstDecoderConfig &,
                    const std::string &, int32_t, const std::string &, float,
                    float, const std::string &, const std::string &,
                    const HomophoneReplacerConfig &>(),
           py::arg("feat_config") = FeatureExtractorConfig(),
           py::arg("model_config") = OfflineModelConfig(),
           py::arg("lm_config") = OfflineLMConfig(),
           py::arg("ctc_fst_decoder_config") = OfflineCtcFstDecoderConfig(),
           py::arg("decoding_method") = "greedy_search",
           py::arg("max_active_paths") = 4, py::arg("hotwords_file") = "",
           py::arg("hotwords_score") = 1.5, py::arg("blank_penalty") = 0.0,
           py::arg("rule_fsts") = "", py::arg("rule_fars") = "",
           py::arg("hr") = HomophoneReplacerConfig{})
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("lm_config", &PyClass::lm_config)
      .def_readwrite("ctc_fst_decoder_config", &PyClass::ctc_fst_decoder_config)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def_readwrite("hotwords_file", &PyClass::hotwords_file)
      .def_readwrite("hotwords_score", &PyClass::hotwords_score)
      .def_readwrite("blank_penalty", &PyClass::blank_penalty)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def_readwrite("rule_fars", &PyClass::rule_fars)
      .def_readwrite("hr", &PyClass::hr)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineRecognizer(py::module *m) {
  PybindOfflineRecognizerConfig(m);

  using PyClass = OfflineRecognizer;
  py::class_<PyClass>(*m, "OfflineRecognizer")
      .def(py::init<const OfflineRecognizerConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](const PyClass &self) { return self.CreateStream(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](PyClass &self, const std::string &hotwords) {
            return self.CreateStream(hotwords);
          },
          py::arg("hotwords"), py::call_guard<py::gil_scoped_release>())
      .def("decode_stream", &PyClass::DecodeStream, py::arg("s"),
           py::call_guard<py::gil_scoped_release>())
      .def("set_config", &PyClass::SetConfig, py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "decode_streams",
          [](const PyClass &self, std::vector<OfflineStream *> ss) {
            self.DecodeStreams(ss.data(), ss.size());
          },
          py::arg("ss"), py::call_guard<py::gil_scoped_release>())
      .def(
          "set_run_options_config_entry",
          [](PyClass &self, const std::string &key, const std::string &value) {
            self.SetRunOptionsConfigEntry(key, value);
          },
          py::arg("key"), py::arg("value"),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Inject a single ONNX Runtime RunOptions config entry that will be
applied to every subsequent decode_stream / decode_streams call.

Example: enable per-Run CPU memory arena shrinkage to keep RSS bounded
under varying input shapes:

    recognizer.set_run_options_config_entry(
        "memory.enable_memory_arena_shrinkage", "cpu:0")

Currently honored by the transducer recognizer. Other recognizer types
log a warning and ignore the setting.

Thread-safety: intended to be called once after recognizer construction
and before the first decode. Concurrent calls with decode are not
synchronized.
)doc")
      .def(
          "set_run_options",
          [](PyClass &self, const py::dict &options) {
            for (auto item : options) {
              self.SetRunOptionsConfigEntry(
                  py::cast<std::string>(item.first),
                  py::cast<std::string>(item.second));
            }
          },
          py::arg("options"),
          R"doc(Convenience wrapper around set_run_options_config_entry that
accepts a dict[str, str] of entries.

    recognizer.set_run_options({
        "memory.enable_memory_arena_shrinkage": "cpu:0",
    })
)doc");
}

}  // namespace sherpa_onnx
