// sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model-config.cc
//
// Copyright (c)  2026  Arbaz Siddiqui

#include "sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSpeechDenoiserVoxLockModelConfig::Register(ParseOptions *po) {
  po->Register("speech-denoiser-voxlock-enroll-model", &enroll_model,
               "Path to the VoxLock enroll model, e.g., voxlock_enroll.onnx. "
               "It runs once on a voice sample of the caller. Download the "
               "models from "
               "https://github.com/arbazsiddiqui/voxlock/releases/tag/"
               "sherpa-onnx-models");

  po->Register("speech-denoiser-voxlock-chunk-model", &chunk_model,
               "Path to the VoxLock chunk model, e.g., "
               "voxlock_chunk_int8.onnx or voxlock_chunk.onnx. It runs on "
               "each chunk of the audio.");

  po->Register(
      "speech-denoiser-voxlock-voice-sample", &voice_sample,
      "Optional. Path to a wave file with at least 3 seconds of the caller's "
      "voice. VoxLock keeps this voice and removes the others. From a longer "
      "file, it uses the loudest 3 seconds within the first 30 seconds. A "
      "file that is shorter or cannot be read is ignored. If empty or "
      "ignored, the offline denoiser uses the loudest 3 seconds in the first "
      "15 seconds of the input, and the online denoiser passes audio through "
      "unchanged until it has heard 3 seconds of speech and then uses that "
      "speech.");
}

bool OfflineSpeechDenoiserVoxLockModelConfig::Validate() const {
  if (enroll_model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --speech-denoiser-voxlock-enroll-model");
    return false;
  }

  if (!FileExists(enroll_model)) {
    SHERPA_ONNX_LOGE("voxlock enroll model file '%s' does not exist",
                     enroll_model.c_str());
    return false;
  }

  if (chunk_model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --speech-denoiser-voxlock-chunk-model");
    return false;
  }

  if (!FileExists(chunk_model)) {
    SHERPA_ONNX_LOGE("voxlock chunk model file '%s' does not exist",
                     chunk_model.c_str());
    return false;
  }

  return true;
}

std::string OfflineSpeechDenoiserVoxLockModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeechDenoiserVoxLockModelConfig(";
  os << "enroll_model=\"" << enroll_model << "\", ";
  os << "chunk_model=\"" << chunk_model << "\", ";
  os << "voice_sample=\"" << voice_sample << "\")";
  return os.str();
}

}  // namespace sherpa_onnx
