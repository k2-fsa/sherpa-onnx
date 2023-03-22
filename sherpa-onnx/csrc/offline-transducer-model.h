// sherpa-onnx/csrc/offline-transducer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_H_

#include <memory>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-transducer-model-config.h"

namespace sherpa_onnx {

class OfflineTransducerModel {
 public:
  explicit OfflineTransducerModel(const OfflineTransducerModelConfig &config);
  ~OfflineTransducerModel();

  /** Run the encoder.
   *
   * @param features  A tensor of shape (N, T, C). It is changed in-place.
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *
   * @return Return a pair containing:
   *  - encoder_out: A 3-D tensor of shape (N, T', encoder_dim)
   *  - encoder_out_length: A 1-D tensor of shape (N,) containing number
   *                        of frames in `encoder_out` before padding.
   */
  std::pair<Ort::Value, Ort::Value> RunEncoder(Ort::Value features,
                                               Ort::Value features_length);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_H_
