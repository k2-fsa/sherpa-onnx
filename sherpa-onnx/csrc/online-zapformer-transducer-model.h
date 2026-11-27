// sherpa-onnx/csrc/online-zapformer-transducer-model.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_ZAPFORMER_TRANSDUCER_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_ZAPFORMER_TRANSDUCER_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"

namespace sherpa_onnx {

// Streaming transducer for models exported by
// icefall/egs/librispeech/ASR/zapformer/export-onnx-streaming.py
//
// The encoder takes x of shape (N, T, feature_dim) followed by the states
// (9 tensors per encoder layer, then embed_cache and processed_lens) and
// returns encoder_out followed by the new states in the same order.
//
// Unlike zipformer2, the state layout is not hard-coded here: the shape,
// element type and batch axis of every state are read from the encoder's
// input type info (the batch axis is the one dynamic dim). All states start
// at zero except cached_norm_stats_i / cached_norm_len_i, which start from
// the learned values in the metadata (norm_init_stats / norm_init_len).
class OnlineZapformerTransducerModel : public OnlineTransducerModel {
 public:
  explicit OnlineZapformerTransducerModel(const OnlineModelConfig &config);

  template <typename Manager>
  OnlineZapformerTransducerModel(Manager *mgr, const OnlineModelConfig &config);

  std::vector<Ort::Value> StackStates(
      const std::vector<std::vector<Ort::Value>> &states) const override;

  std::vector<std::vector<Ort::Value>> UnStackStates(
      const std::vector<Ort::Value> &states) const override;

  std::vector<Ort::Value> GetEncoderInitStates() override;

  void SetFeatureDim(int32_t feature_dim) override {
    feature_dim_ = feature_dim;
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> RunEncoder(
      Ort::Value features, std::vector<Ort::Value> states,
      Ort::Value processed_frames) override;

  Ort::Value RunDecoder(Ort::Value decoder_input) override;

  Ort::Value RunJoiner(Ort::Value encoder_out, Ort::Value decoder_out) override;

  int32_t ContextSize() const override { return context_size_; }

  int32_t ChunkSize() const override { return T_; }

  int32_t ChunkShift() const override { return decode_chunk_len_; }

  int32_t VocabSize() const override { return vocab_size_; }
  OrtAllocator *Allocator() override { return allocator_; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length);
  void InitDecoder(void *model_data, size_t model_data_length);
  void InitJoiner(void *model_data, size_t model_data_length);

  struct StateInfo {
    std::string name;
    std::vector<int64_t> shape;  // batch axis set to 1
    ONNXTensorElementDataType type;
    int32_t batch_axis;
    float init_value;
  };

 private:
  Ort::Env env_;
  Ort::SessionOptions encoder_sess_opts_;
  Ort::SessionOptions decoder_sess_opts_;
  Ort::SessionOptions joiner_sess_opts_;

  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;
  std::unique_ptr<Ort::Session> joiner_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<std::string> joiner_input_names_;
  std::vector<const char *> joiner_input_names_ptr_;

  std::vector<std::string> joiner_output_names_;
  std::vector<const char *> joiner_output_names_ptr_;

  OnlineModelConfig config_;

  std::vector<StateInfo> states_info_;  // encoder inputs 1..end

  int32_t T_ = 0;
  int32_t decode_chunk_len_ = 0;

  int32_t context_size_ = 0;
  int32_t vocab_size_ = 0;
  int32_t feature_dim_ = 80;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_ZAPFORMER_TRANSDUCER_MODEL_H_
