// sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_RKNN_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_RKNN_H_
#define SHERPA_ONNX_CSRC_RKNN_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_RKNN_H_

#include <memory>
#include <utility>
#include <vector>

#include "rknn_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"

namespace sherpa_onnx {

// this is for zipformer v1 and v2, i.e., the folder
// pruned_transducer_statelss7_streaming
// and
// zipformer
// from icefall
class OnlineZipformerTransducerModelRknn {
 public:
  ~OnlineZipformerTransducerModelRknn();

  explicit OnlineZipformerTransducerModelRknn(const OnlineModelConfig &config);

  template <typename Manager>
  OnlineZipformerTransducerModelRknn(Manager *mgr,
                                     const OnlineModelConfig &config);

  std::vector<std::vector<uint8_t>> GetEncoderInitStates() const;

  std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>> RunEncoder(
      std::vector<float> features,
      std::vector<std::vector<uint8_t>> states) const;

  std::vector<float> RunDecoder(std::vector<int64_t> decoder_input) const;

  std::vector<float> RunJoiner(const float *encoder_out,
                               const float *decoder_out) const;

  int32_t ContextSize() const;

  int32_t ChunkSize() const;

  int32_t ChunkShift() const;

  int32_t VocabSize() const;

  rknn_tensor_attr GetEncoderOutAttr() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_RKNN_H_
