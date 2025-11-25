// sherpa-onnx/csrc/ascend/offline-zipformer-ctc-model-ascend.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ASCEND_OFFLINE_ZIPFORMER_CTC_MODEL_ASCEND_H_
#define SHERPA_ONNX_CSRC_ASCEND_OFFLINE_ZIPFORMER_CTC_MODEL_ASCEND_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineZipformerCtcModelAscend {
 public:
  ~OfflineZipformerCtcModelAscend();

  explicit OfflineZipformerCtcModelAscend(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineZipformerCtcModelAscend(Manager *mgr,
                                 const OfflineModelConfig &config);

  /**
   * @param features A tensor of shape (num_frames, feature_dim)
   * @returns Return a tensor of shape (num_output_frames, vocab_size)
   */
  std::vector<float> Run(std::vector<float> features) const;

  int32_t VocabSize() const;
  int32_t SubsamplingFactor() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_OFFLINE_ZIPFORMER_CTC_MODEL_ASCEND_H_
