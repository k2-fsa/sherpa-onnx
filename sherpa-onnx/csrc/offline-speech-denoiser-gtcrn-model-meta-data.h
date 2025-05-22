// sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-model-meta-data.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_META_DATA_H_

#include <cstdint>
#include <string>
#include <vector>

namespace sherpa_onnx {

// please refer to
// https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/gtcrn/add_meta_data.py
struct OfflineSpeechDenoiserGtcrnModelMetaData {
  int32_t sample_rate = 0;
  int32_t version = 1;
  int32_t n_fft = 0;
  int32_t hop_length = 0;
  int32_t window_length = 0;
  std::string window_type;

  std::vector<int64_t> conv_cache_shape;
  std::vector<int64_t> tra_cache_shape;
  std::vector<int64_t> inter_cache_shape;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_META_DATA_H_
