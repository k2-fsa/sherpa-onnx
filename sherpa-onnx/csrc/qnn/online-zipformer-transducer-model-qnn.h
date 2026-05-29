// sherpa-onnx/csrc/qnn/online-zipformer-transducer-model-qnn.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_QNN_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_QNN_H_
#define SHERPA_ONNX_CSRC_QNN_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_QNN_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-stream-state.h"

namespace sherpa_onnx {

class OnlineZipformerTransducerModelQnn {
 public:
  ~OnlineZipformerTransducerModelQnn();

  OnlineZipformerTransducerModelQnn(const OnlineModelConfig &config,
                                    int32_t feature_dim);

  template <typename Manager>
  OnlineZipformerTransducerModelQnn(Manager *mgr,
                                    const OnlineModelConfig &config,
                                    int32_t feature_dim);

  std::vector<OnlineStreamStateTensor> GetEncoderInitStates() const;

  std::vector<float> RunEncoder(std::vector<float> features, int32_t num_frames,
                               std::vector<OnlineStreamStateTensor> *states) const;

  std::vector<float> RunDecoder(const std::vector<int32_t> &tokens) const;

  std::vector<float> RunJoiner(const float *encoder_out,
                               const std::vector<float> &decoder_out) const;

  int32_t ContextSize() const;
  int32_t ChunkSize() const;
  int32_t ChunkShift() const;
  int32_t VocabSize() const;
  int32_t FeatureDim() const;
  int32_t EncoderDim() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_QNN_H_
