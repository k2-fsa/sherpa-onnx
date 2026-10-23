// sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model-meta-data.h
//
// Copyright (c)  2026  Arbaz Siddiqui

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_MODEL_META_DATA_H_

#include <cstdint>
#include <string>
#include <vector>

namespace sherpa_onnx {

// please refer to
// https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/voxlock/add_meta_data.py
struct OfflineSpeechDenoiserVoxLockModelMetaData {
  int32_t version = 1;
  int32_t sample_rate = 0;
  int32_t n_fft = 0;
  int32_t hop_length = 0;
  int32_t window_length = 0;
  std::string window_type;

  // From the enroll model
  int32_t min_voice_sample_length = 0;
  int32_t voice_sample_seconds = 0;

  // From the chunk model
  int32_t frames_per_chunk = 0;
  float output_gain = 0;
  std::vector<int64_t> state_shape;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_VOXLOCK_MODEL_META_DATA_H_
