// sherpa-onnx/python/csrc/speaker-embedding-manager.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/speaker-embedding-manager.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/speaker-embedding-manager.h"

namespace sherpa_onnx {

static constexpr const char *kSpeakerEmbeddingManagerInitDoc = R"doc(
Constructor for SpeakerEmbeddingManager.

Args:
  dim:
    Dimension of the embedding vector.
)doc";

static constexpr const char *kSpeakerEmbeddingManagerNumSpeakersDoc = R"doc(
Return the number of registered speakers.
)doc";

static constexpr const char *kSpeakerEmbeddingManagerDimDoc = R"doc(
Return the dimension of the embedding vector.
)doc";

static constexpr const char *kSpeakerEmbeddingManagerAllSpeakersDoc = R"doc(
Return a list of all registered speaker names.
)doc";

static constexpr const char *kSpeakerEmbeddingManagerContainsDoc = R"doc(
Return True if the given speaker name is registered.

Args:
  name:
    The speaker name to check.
)doc";

static constexpr const char *kSpeakerEmbeddingManagerAddDoc = R"doc(
Register a speaker with the given embedding or list of embeddings.

Args:
  name:
    The speaker name.
  v:
    A 1-D float32 array representing the speaker embedding.
  embedding_list:
    A list of 1-D float32 arrays representing multiple embeddings.

Returns:
  True if the speaker was added successfully.
)doc";

static constexpr const char *kSpeakerEmbeddingManagerRemoveDoc = R"doc(
Remove a registered speaker.

Args:
  name:
    The speaker name to remove.

Returns:
  True if the speaker was removed successfully.
)doc";

static constexpr const char *kSpeakerEmbeddingManagerSearchDoc = R"doc(
Search for the speaker whose embedding is closest to the given one.

Args:
  v:
    A 1-D float32 array representing the query embedding.
  threshold:
    The similarity threshold.

Returns:
  The matched speaker name, or an empty string if no match.
)doc";

static constexpr const char *kSpeakerEmbeddingManagerVerifyDoc = R"doc(
Verify whether the given embedding matches the registered speaker.

Args:
  name:
    The speaker name to verify against.
  v:
    A 1-D float32 array representing the query embedding.
  threshold:
    The similarity threshold.

Returns:
  True if the embedding matches the speaker.
)doc";

static constexpr const char *kSpeakerEmbeddingManagerScoreDoc = R"doc(
Compute the similarity score between the given embedding and a speaker.

Args:
  name:
    The speaker name.
  v:
    A 1-D float32 array representing the query embedding.

Returns:
  The similarity score.
)doc";

void PybindSpeakerEmbeddingManager(py::module *m) {
  using PyClass = SpeakerEmbeddingManager;
  py::class_<PyClass>(*m, "SpeakerEmbeddingManager")
      .def(py::init<int32_t>(), py::arg("dim"),
           py::call_guard<py::gil_scoped_release>(),
           kSpeakerEmbeddingManagerInitDoc)
      .def_property_readonly("num_speakers", &PyClass::NumSpeakers,
                             kSpeakerEmbeddingManagerNumSpeakersDoc)
      .def_property_readonly("dim", &PyClass::Dim,
                             kSpeakerEmbeddingManagerDimDoc)
      .def_property_readonly("all_speakers", &PyClass::GetAllSpeakers,
                             kSpeakerEmbeddingManagerAllSpeakersDoc)
      .def(
          "__contains__",
          [](const PyClass &self, const std::string &name) -> bool {
            return self.Contains(name);
          },
          py::arg("name"), py::call_guard<py::gil_scoped_release>(),
          kSpeakerEmbeddingManagerContainsDoc)
      .def(
          "add",
          [](const PyClass &self, const std::string &name,
             const std::vector<float> &v) -> bool {
            return self.Add(name, v.data());
          },
          py::arg("name"), py::arg("v"),
          py::call_guard<py::gil_scoped_release>(),
          kSpeakerEmbeddingManagerAddDoc)
      .def(
          "add",
          [](const PyClass &self, const std::string &name,
             const std::vector<std::vector<float>> &embedding_list) -> bool {
            return self.Add(name, embedding_list);
          },
          py::arg("name"), py::arg("embedding_list"),
          py::call_guard<py::gil_scoped_release>(),
          kSpeakerEmbeddingManagerAddDoc)
      .def(
          "remove",
          [](const PyClass &self, const std::string &name) -> bool {
            return self.Remove(name);
          },
          py::arg("name"), py::call_guard<py::gil_scoped_release>(),
          kSpeakerEmbeddingManagerRemoveDoc)
      .def(
          "search",
          [](const PyClass &self, const std::vector<float> &v, float threshold)
              -> std::string { return self.Search(v.data(), threshold); },
          py::arg("v"), py::arg("threshold"),
          py::call_guard<py::gil_scoped_release>(),
          kSpeakerEmbeddingManagerSearchDoc)
      .def(
          "verify",
          [](const PyClass &self, const std::string &name,
             const std::vector<float> &v, float threshold) -> bool {
            return self.Verify(name, v.data(), threshold);
          },
          py::arg("name"), py::arg("v"), py::arg("threshold"),
          py::call_guard<py::gil_scoped_release>(),
          kSpeakerEmbeddingManagerVerifyDoc)
      .def(
          "score",
          [](const PyClass &self, const std::string &name,
             const std::vector<float> &v) -> float {
            return self.Score(name, v.data());
          },
          py::arg("name"), py::arg("v"),
          py::call_guard<py::gil_scoped_release>(),
          kSpeakerEmbeddingManagerScoreDoc);
}

}  // namespace sherpa_onnx
