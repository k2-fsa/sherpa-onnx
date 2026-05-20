// sherpa-onnx/python/csrc/keyword-spotter.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/keyword-spotter.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/keyword-spotter.h"

namespace sherpa_onnx {

static void PybindKeywordResult(py::module *m) {
  using PyClass = KeywordResult;
  py::class_<PyClass>(*m, "KeywordResult")
      .def_property_readonly(
          "keyword",
          [](PyClass &self) -> py::str {
            return py::str(PyUnicode_DecodeUTF8(self.keyword.c_str(),
                                                self.keyword.size(), "ignore"));
          })
      .def_property_readonly(
          "tokens",
          [](PyClass &self) -> std::vector<std::string> { return self.tokens; })
      .def_property_readonly(
          "timestamps",
          [](PyClass &self) -> std::vector<float> { return self.timestamps; });
}

static void PybindKeywordSpotterConfig(py::module *m) {
  using PyClass = KeywordSpotterConfig;
  py::class_<PyClass>(*m, "KeywordSpotterConfig")
      .def(py::init<const FeatureExtractorConfig &, const OnlineModelConfig &,
                    int32_t, int32_t, float, float, const std::string &>(),
           py::arg("feat_config"), py::arg("model_config"),
           py::arg("max_active_paths") = 4, py::arg("num_trailing_blanks") = 1,
           py::arg("keywords_score") = 1.0,
           py::arg("keywords_threshold") = 0.25, py::arg("keywords_file") = "")
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def_readwrite("num_trailing_blanks", &PyClass::num_trailing_blanks)
      .def_readwrite("keywords_score", &PyClass::keywords_score)
      .def_readwrite("keywords_threshold", &PyClass::keywords_threshold)
      .def_readwrite("keywords_file", &PyClass::keywords_file)
      .def("__str__", &PyClass::ToString);
}

static constexpr const char *kKeywordSpotterDoc = R"doc(
Keyword spotter engine.

Args:
  config:
    The configuration for the keyword spotter.
)doc";

static constexpr const char *kCreateStreamDoc = R"doc(
Create a new streaming recognition instance using the keywords from the
configuration.

Returns:
  An ``OnlineStream`` object.
)doc";

static constexpr const char *kCreateStreamKeywordsDoc = R"doc(
Create a new streaming recognition instance with custom keywords.

Args:
  keywords:
    A string of keywords to spot, separated by ``/``.

Returns:
  An ``OnlineStream`` object.
)doc";

static constexpr const char *kIsReadyDoc = R"doc(
Return True if the stream has enough frames for decoding.

Args:
  stream:
    The stream to check.
)doc";

static constexpr const char *kDecodeStreamDoc = R"doc(
Run one decoding step on the given stream.

Args:
  stream:
    The stream to decode.
)doc";

static constexpr const char *kGetResultDoc = R"doc(
Get the current keyword result for the given stream.

Args:
  stream:
    The stream to query.

Returns:
  A ``KeywordResult`` object.
)doc";

static constexpr const char *kResetStreamDoc = R"doc(
Reset the given stream for the next detection round.

Args:
  stream:
    The stream to reset.
)doc";

void PybindKeywordSpotter(py::module *m) {
  PybindKeywordResult(m);
  PybindKeywordSpotterConfig(m);

  using PyClass = KeywordSpotter;
  py::class_<PyClass>(*m, "KeywordSpotter", kKeywordSpotterDoc)
      .def(py::init<const KeywordSpotterConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](const PyClass &self) { return self.CreateStream(); },
          kCreateStreamDoc, py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](PyClass &self, const std::string &keywords) {
            return self.CreateStream(keywords);
          },
          py::arg("keywords"), kCreateStreamKeywordsDoc,
          py::call_guard<py::gil_scoped_release>())
      .def("is_ready", &PyClass::IsReady, kIsReadyDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, kResetStreamDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("decode_stream", &PyClass::DecodeStream, kDecodeStreamDoc,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "decode_streams",
          [](PyClass &self, std::vector<OnlineStream *> ss) {
            self.DecodeStreams(ss.data(), ss.size());
          },
          py::call_guard<py::gil_scoped_release>())
      .def("get_result", &PyClass::GetResult, kGetResultDoc,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_onnx
