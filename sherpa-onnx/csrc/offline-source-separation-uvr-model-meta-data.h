// sherpa-onnx/csrc/offline-source-separation-uvr-model-meta-data.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_MODEL_META_DATA_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_onnx {

// See also
// https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/uvr_mdx/test.py
// https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/uvr_mdx/add_meta_data_and_quantize.py
struct OfflineSourceSeparationUvrModelMetaData {
  int32_t sample_rate = 44100;
  int32_t num_stems = 2;
  int32_t dim_c = -1;
  int32_t dim_f = -1;
  int32_t dim_t = -1;

  int32_t n_fft = -1;
  int32_t hop_length = 1024;

  int32_t window_length = -1;
  int32_t center = 1;
  std::string window_type = "hann";

  // the following fields are preconfigured. Please see
  // https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/uvr_mdx/test.py
  int32_t margin = 0;  // changed in ./offline-source-separation-uvr-model.cc
  const int32_t num_chunks = 15;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_MODEL_META_DATA_H_
