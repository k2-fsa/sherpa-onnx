// sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model-meta-data.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_MODEL_META_DATA_H_

#include <cstdint>
#include <string>
#include <vector>

namespace sherpa_onnx {

struct OfflineSpeechDenoiserDpdfNetModelMetaData {
  int32_t sample_rate = 0;
  int32_t n_fft = 0;
  int32_t hop_length = 0;
  int32_t window_length = 0;
  bool normalized = false;
  bool center = true;
  std::string window_type = "vorbis";
  std::string pad_mode = "reflect";
  int32_t state_size = 0;
  std::string profile;

  std::vector<int64_t> spec_shape;
  std::vector<int64_t> state_shape;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_MODEL_META_DATA_H_
