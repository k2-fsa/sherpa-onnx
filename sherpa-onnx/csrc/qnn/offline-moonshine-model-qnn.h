// sherpa-onnx/csrc/qnn/offline-moonshine-model-qnn.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_QNN_OFFLINE_MOONSHINE_MODEL_QNN_H_
#define SHERPA_ONNX_CSRC_QNN_OFFLINE_MOONSHINE_MODEL_QNN_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-moonshine-decoder.h"

namespace sherpa_onnx {

class OfflineMoonshineModelQnn {
 public:
  ~OfflineMoonshineModelQnn();

  explicit OfflineMoonshineModelQnn(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineMoonshineModelQnn(Manager *mgr, const OfflineModelConfig &config);

  /** Run the model on raw audio samples.
   *
   * @param audio_samples float32 audio at 16kHz
   * @return decoder result with token ids
   */
  OfflineMoonshineDecoderResult Run(const std::vector<float> &audio_samples) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_OFFLINE_MOONSHINE_MODEL_QNN_H_
