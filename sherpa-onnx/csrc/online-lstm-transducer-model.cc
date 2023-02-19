// sherpa/csrc/online-lstm-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-lstm-transducer-model.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

#define SHERPA_ONNX_READ_META_DATA(dst, src_key)                        \
  do {                                                                  \
    auto value =                                                        \
        meta_data.LookupCustomMetadataMapAllocated(src_key, allocator); \
    if (!value) {                                                       \
      fprintf(stderr, "%s does not exist in the metadata\n", src_key);  \
      exit(-1);                                                         \
    }                                                                   \
    dst = atoi(value.get());                                            \
    if (dst <= 0) {                                                     \
      fprintf(stderr, "Invalud value %d for %s\n", dst, src_key);       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

namespace sherpa_onnx {

OnlineLstmTransducerModel::OnlineLstmTransducerModel(
    const OnlineTransducerModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_WARNING),
      config_(config),
      sess_opts_{},
      allocator_{} {
  sess_opts_.SetIntraOpNumThreads(config.num_threads);
  sess_opts_.SetInterOpNumThreads(config.num_threads);

  InitEncoder(config.encoder_filename);
  InitDecoder(config.decoder_filename);
  InitJoiner(config.joiner_filename);
}

void OnlineLstmTransducerModel::InitEncoder(const std::string &filename) {
  encoder_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                &encoder_input_names_ptr_);

  GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                 &encoder_output_names_ptr_);

  // get meta data
  Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
  if (config_.debug) {
    std::ostringstream os;
    os << "---encoder---\n";
    PrintModelMetadata(os, meta_data);
    fprintf(stderr, "%s\n", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;
  SHERPA_ONNX_READ_META_DATA(num_encoder_layers_, "num_encoder_layers");
  SHERPA_ONNX_READ_META_DATA(T_, "T");
  SHERPA_ONNX_READ_META_DATA(decode_chunk_len_, "decode_chunk_len");
  SHERPA_ONNX_READ_META_DATA(rnn_hidden_size_, "rnn_hidden_size");
  SHERPA_ONNX_READ_META_DATA(d_model_, "d_model");
}

void OnlineLstmTransducerModel::InitDecoder(const std::string &filename) {
  decoder_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                &decoder_input_names_ptr_);

  GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                 &decoder_output_names_ptr_);

  // get meta data
  Ort::ModelMetadata meta_data = decoder_sess_->GetModelMetadata();
  if (config_.debug) {
    std::ostringstream os;
    os << "---decoder---\n";
    PrintModelMetadata(os, meta_data);
    fprintf(stderr, "%s\n", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;
  SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
  SHERPA_ONNX_READ_META_DATA(context_size_, "context_size");
}

void OnlineLstmTransducerModel::InitJoiner(const std::string &filename) {
  joiner_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(joiner_sess_.get(), &joiner_input_names_,
                &joiner_input_names_ptr_);

  GetOutputNames(joiner_sess_.get(), &joiner_output_names_,
                 &joiner_output_names_ptr_);

  // get meta data
  Ort::ModelMetadata meta_data = joiner_sess_->GetModelMetadata();
  if (config_.debug) {
    std::ostringstream os;
    os << "---joiner---\n";
    PrintModelMetadata(os, meta_data);
    fprintf(stderr, "%s\n", os.str().c_str());
  }
}

std::vector<Ort::Value> OnlineLstmTransducerModel::StackStates(
    const std::vector<std::vector<Ort::Value>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());

  std::array<int64_t, 3> h_shape{num_encoder_layers_, batch_size, d_model_};
  Ort::Value h = Ort::Value::CreateTensor<float>(allocator_, h_shape.data(),
                                                 h_shape.size());

  std::array<int64_t, 3> c_shape{num_encoder_layers_, batch_size,
                                 rnn_hidden_size_};

  Ort::Value c = Ort::Value::CreateTensor<float>(allocator_, c_shape.data(),
                                                 c_shape.size());

  float *dst_h = h.GetTensorMutableData<float>();
  float *dst_c = c.GetTensorMutableData<float>();

  for (int32_t layer = 0; layer != num_encoder_layers_; ++layer) {
    for (int32_t i = 0; i != batch_size; ++i) {
      const float *src_h =
          states[i][0].GetTensorData<float>() + layer * d_model_;

      const float *src_c =
          states[i][1].GetTensorData<float>() + layer * rnn_hidden_size_;

      std::copy(src_h, src_h + d_model_, dst_h);
      std::copy(src_c, src_c + rnn_hidden_size_, dst_c);

      dst_h += d_model_;
      dst_c += rnn_hidden_size_;
    }
  }

  std::vector<Ort::Value> ans;

  ans.reserve(2);
  ans.push_back(std::move(h));
  ans.push_back(std::move(c));

  return ans;
}

std::vector<std::vector<Ort::Value>> OnlineLstmTransducerModel::UnStackStates(
    const std::vector<Ort::Value> &states) const {
  int32_t batch_size = states[0].GetTensorTypeAndShapeInfo().GetShape()[1];

  std::vector<std::vector<Ort::Value>> ans(batch_size);

  // allocate space
  std::array<int64_t, 3> h_shape{num_encoder_layers_, 1, d_model_};
  std::array<int64_t, 3> c_shape{num_encoder_layers_, 1, rnn_hidden_size_};

  for (int32_t i = 0; i != batch_size; ++i) {
    Ort::Value h = Ort::Value::CreateTensor<float>(allocator_, h_shape.data(),
                                                   h_shape.size());
    Ort::Value c = Ort::Value::CreateTensor<float>(allocator_, c_shape.data(),
                                                   c_shape.size());
    ans[i].push_back(std::move(h));
    ans[i].push_back(std::move(c));
  }

  for (int32_t layer = 0; layer != num_encoder_layers_; ++layer) {
    for (int32_t i = 0; i != batch_size; ++i) {
      const float *src_h = states[0].GetTensorData<float>() +
                           layer * batch_size * d_model_ + i * d_model_;
      const float *src_c = states[1].GetTensorData<float>() +
                           layer * batch_size * rnn_hidden_size_ +
                           i * rnn_hidden_size_;

      float *dst_h = ans[i][0].GetTensorMutableData<float>() + layer * d_model_;
      float *dst_c =
          ans[i][1].GetTensorMutableData<float>() + layer * rnn_hidden_size_;

      std::copy(src_h, src_h + d_model_, dst_h);
      std::copy(src_c, src_c + rnn_hidden_size_, dst_c);
    }
  }

  return ans;
}

std::vector<Ort::Value> OnlineLstmTransducerModel::GetEncoderInitStates() {
  // Please see
  // https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/lstm_transducer_stateless2/export-onnx.py#L185
  // for details
  constexpr int32_t kBatchSize = 1;
  std::array<int64_t, 3> h_shape{num_encoder_layers_, kBatchSize, d_model_};
  Ort::Value h = Ort::Value::CreateTensor<float>(allocator_, h_shape.data(),
                                                 h_shape.size());

  std::fill(h.GetTensorMutableData<float>(),
            h.GetTensorMutableData<float>() +
                num_encoder_layers_ * kBatchSize * d_model_,
            0);

  std::array<int64_t, 3> c_shape{num_encoder_layers_, kBatchSize,
                                 rnn_hidden_size_};
  Ort::Value c = Ort::Value::CreateTensor<float>(allocator_, c_shape.data(),
                                                 c_shape.size());

  std::fill(c.GetTensorMutableData<float>(),
            c.GetTensorMutableData<float>() +
                num_encoder_layers_ * kBatchSize * rnn_hidden_size_,
            0);

  std::vector<Ort::Value> states;

  states.reserve(2);
  states.push_back(std::move(h));
  states.push_back(std::move(c));

  return states;
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineLstmTransducerModel::RunEncoder(Ort::Value features,
                                      std::vector<Ort::Value> &states) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::array<Ort::Value, 3> encoder_inputs = {
      std::move(features), std::move(states[0]), std::move(states[1])};

  auto encoder_out = encoder_sess_->Run(
      {}, encoder_input_names_ptr_.data(), encoder_inputs.data(),
      encoder_inputs.size(), encoder_output_names_ptr_.data(),
      encoder_output_names_ptr_.size());

  std::vector<Ort::Value> next_states;
  next_states.reserve(2);
  next_states.push_back(std::move(encoder_out[1]));
  next_states.push_back(std::move(encoder_out[2]));

  return {std::move(encoder_out[0]), std::move(next_states)};
}

Ort::Value OnlineLstmTransducerModel::BuildDecoderInput(
    const std::vector<OnlineTransducerDecoderResult> &results) {
  int32_t batch_size = static_cast<int32_t>(results.size());
  std::array<int64_t, 2> shape{batch_size, context_size_};
  Ort::Value decoder_input =
      Ort::Value::CreateTensor<int64_t>(allocator_, shape.data(), shape.size());
  int64_t *p = decoder_input.GetTensorMutableData<int64_t>();

  for (const auto &r : results) {
    const int64_t *begin = r.tokens.data() + r.tokens.size() - context_size_;
    const int64_t *end = r.tokens.data() + r.tokens.size();
    std::copy(begin, end, p);
    p += context_size_;
  }

  return decoder_input;
}

Ort::Value OnlineLstmTransducerModel::RunDecoder(Ort::Value decoder_input) {
  auto decoder_out = decoder_sess_->Run(
      {}, decoder_input_names_ptr_.data(), &decoder_input, 1,
      decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());
  return std::move(decoder_out[0]);
}

Ort::Value OnlineLstmTransducerModel::RunJoiner(Ort::Value encoder_out,
                                                Ort::Value decoder_out) {
  std::array<Ort::Value, 2> joiner_input = {std::move(encoder_out),
                                            std::move(decoder_out)};
  auto logit =
      joiner_sess_->Run({}, joiner_input_names_ptr_.data(), joiner_input.data(),
                        joiner_input.size(), joiner_output_names_ptr_.data(),
                        joiner_output_names_ptr_.size());

  return std::move(logit[0]);
}

}  // namespace sherpa_onnx
