// sherpa-onnx/python/csrc/speaker-embedding-extractor.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/speaker-embedding-extractor.h"

#include <string>

#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

namespace sherpa_onnx {

static constexpr const char *kSpeakerEmbeddingExtractorInitDoc = R"doc(
Constructor for SpeakerEmbeddingExtractor.

Args:
  config:
    Config for speaker embedding extractor.
)doc";

static constexpr const char *kSpeakerEmbeddingExtractorDimDoc = R"doc(
Return the dimension of the embedding vector.
)doc";

static constexpr const char *kSpeakerEmbeddingExtractorCreateStreamDoc = R"doc(
Create a stream for feeding audio data.

Returns:
  An OnlineStream object.
)doc";

static constexpr const char *kSpeakerEmbeddingExtractorComputeDoc = R"doc(
Compute the speaker embedding from the given stream.

Returns:
  A 1-D float32 numpy array of the embedding.
)doc";

static constexpr const char *kSpeakerEmbeddingExtractorIsReadyDoc = R"doc(
Return True if the stream has enough audio data for embedding extraction.
)doc";

static void PybindSpeakerEmbeddingExtractorConfig(py::module *m) {
  using PyClass = SpeakerEmbeddingExtractorConfig;
  py::class_<PyClass>(*m, "SpeakerEmbeddingExtractorConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, int32_t, bool, const std::string &>(),
           py::arg("model"), py::arg("num_threads") = 1,
           py::arg("debug") = false, py::arg("provider") = "cpu")
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindSpeakerEmbeddingExtractor(py::module *m) {
  PybindSpeakerEmbeddingExtractorConfig(m);

  using PyClass = SpeakerEmbeddingExtractor;
  py::class_<PyClass>(*m, "SpeakerEmbeddingExtractor")
      .def(py::init<const SpeakerEmbeddingExtractorConfig &>(),
           py::arg("config"), py::call_guard<py::gil_scoped_release>(),
           kSpeakerEmbeddingExtractorInitDoc)
      .def_property_readonly("dim", &PyClass::Dim,
                             kSpeakerEmbeddingExtractorDimDoc)
      .def("create_stream", &PyClass::CreateStream,
           py::call_guard<py::gil_scoped_release>(),
           kSpeakerEmbeddingExtractorCreateStreamDoc)
      .def("compute", &PyClass::Compute,
           py::call_guard<py::gil_scoped_release>(),
           kSpeakerEmbeddingExtractorComputeDoc)
      .def("is_ready", &PyClass::IsReady,
           py::call_guard<py::gil_scoped_release>(),
           kSpeakerEmbeddingExtractorIsReadyDoc);
}

}  // namespace sherpa_onnx
