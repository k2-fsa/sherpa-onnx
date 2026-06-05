// sherpa-onnx/csrc/qnn/offline-zipformer-transducer-model-qnn.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_QNN_OFFLINE_ZIPFORMER_TRANSDUCER_MODEL_QNN_H_
#define SHERPA_ONNX_CSRC_QNN_OFFLINE_ZIPFORMER_TRANSDUCER_MODEL_QNN_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineZipformerTransducerModelQnn {
 public:
  ~OfflineZipformerTransducerModelQnn();

  explicit OfflineZipformerTransducerModelQnn(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineZipformerTransducerModelQnn(Manager *mgr,
                                     const OfflineModelConfig &config);

  std::vector<float> RunEncoder(std::vector<float> features) const;

  std::vector<float> RunDecoder(const std::vector<int32_t> &tokens) const;

  std::vector<float> RunJoiner(const float *encoder_out,
                               const std::vector<float> &decoder_out) const;

  int32_t ContextSize() const;
  int32_t VocabSize() const;
  int32_t SubsamplingFactor() const;
  int32_t EncoderDim() const;
  int32_t NumEncoderFrames(int32_t num_frames) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_OFFLINE_ZIPFORMER_TRANSDUCER_MODEL_QNN_H_
