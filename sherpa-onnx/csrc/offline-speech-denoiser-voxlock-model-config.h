// sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model-config.h
//
// Copyright (c)  2026  Arbaz Siddiqui
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// VoxLock keeps the voice of one speaker (the caller) and removes other
// voices. See https://github.com/arbazsiddiqui/voxlock
struct OfflineSpeechDenoiserVoxLockModelConfig {
  // Runs once on a voice sample of the caller
  std::string enroll_model;

  // Runs on each chunk of the audio
  std::string chunk_model;

  // Optional. A wave file with about 3 seconds of the caller's voice.
  // If empty, the voice sample is taken from the input audio: the loudest
  // 3 seconds in the first 15 seconds for the offline denoiser, and the
  // first 3 seconds of speech for the online denoiser, which passes audio
  // through unchanged until then.
  std::string voice_sample;

  OfflineSpeechDenoiserVoxLockModelConfig() = default;

  OfflineSpeechDenoiserVoxLockModelConfig(const std::string &enroll_model,
                                          const std::string &chunk_model,
                                          const std::string &voice_sample)
      : enroll_model(enroll_model),
        chunk_model(chunk_model),
        voice_sample(voice_sample) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_MODEL_CONFIG_H_
