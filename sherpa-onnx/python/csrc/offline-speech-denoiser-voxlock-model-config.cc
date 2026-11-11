// sherpa-onnx/python/csrc/offline-speech-denoiser-voxlock-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/offline-speech-denoiser-voxlock-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model-config.h"

namespace sherpa_onnx {

static constexpr const char *kVoxLockModelConfigDoc = R"doc(
Configuration for the VoxLock speech denoiser.

VoxLock keeps the voice of one speaker (the caller) and removes the other
voices.

The model is from https://github.com/arbazsiddiqui/voxlock
)doc";

static constexpr const char *kVoxLockEnrollModelDoc = R"doc(
Path to the VoxLock enroll model, e.g., ``enroll.onnx``.

It runs once on a voice sample of the speaker to keep. Download the models
from
https://github.com/arbazsiddiqui/voxlock/releases/tag/sherpa-onnx-models
)doc";

static constexpr const char *kVoxLockChunkModelDoc = R"doc(
Path to the VoxLock chunk model, e.g., ``chunk.onnx`` or
``chunk_int8.onnx``.

It runs on each chunk of the audio.
)doc";

static constexpr const char *kVoxLockVoiceSampleDoc = R"doc(
Optional. Path to a wave file with at least 3 seconds of the caller's
voice. VoxLock keeps this voice and removes the others. From a longer
file, it uses the loudest 3 seconds within the first 30 seconds. A file
that is shorter or cannot be read is ignored, as if it were empty.

If empty or ignored, the voice sample is taken from the input audio:

  - Offline denoiser: the loudest 3 seconds in the first 15 seconds of
    the input are used.
  - Online denoiser: the audio is passed through unchanged until 3
    seconds of speech have been heard; that speech is used as the voice
    sample.

Defaults to ``""``.
)doc";

void PybindOfflineSpeechDenoiserVoxLockModelConfig(py::module *m) {
  using PyClass = OfflineSpeechDenoiserVoxLockModelConfig;
  py::class_<PyClass>(*m, "OfflineSpeechDenoiserVoxLockModelConfig",
                      kVoxLockModelConfigDoc)
      .def(py::init<const std::string &, const std::string &,
                    const std::string &>(),
           py::arg("enroll_model") = "", py::arg("chunk_model") = "",
           py::arg("voice_sample") = "")
      .def_readwrite("enroll_model", &PyClass::enroll_model,
                     kVoxLockEnrollModelDoc)
      .def_readwrite("chunk_model", &PyClass::chunk_model,
                     kVoxLockChunkModelDoc)
      .def_readwrite("voice_sample", &PyClass::voice_sample,
                     kVoxLockVoiceSampleDoc)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx
