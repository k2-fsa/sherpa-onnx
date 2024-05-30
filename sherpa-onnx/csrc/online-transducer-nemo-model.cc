// sherpa-onnx/csrc/online-transducer-nemo-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#include "sherpa-onnx/csrc/online-transducer-nemo-model.h"

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transpose.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

class OnlineTransducerNeMoModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.joiner);
      InitJoiner(buf.data(), buf.size());
    }
  }
  
#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.transducer.encoder_filename);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.decoder_filename);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.joiner_filename);
      InitJoiner(buf.data(), buf.size());
    }
  }
#endif

  std::vector<Ort::Value> RunEncoder(Ort::Value features,
                                    std::vector<Ort::Value> states) {
    Ort::Value &cache_last_channel = states[0];
    Ort::Value &cache_last_time = states[1];
    Ort::Value &cache_last_channel_len = states[2];

    int32_t batch_size = features.GetTensorTypeAndShapeInfo().GetShape()[0];

    std::array<int64_t, 1> length_shape{batch_size};

    Ort::Value length = Ort::Value::CreateTensor<int64_t>(
        allocator_, length_shape.data(), length_shape.size());

    int64_t *p_length = length.GetTensorMutableData<int64_t>();

    std::fill(p_length, p_length + batch_size, ChunkSize());

    // (B, T, C) -> (B, C, T)
    features = Transpose12(allocator_, &features);

    std::array<Ort::Value, 5> inputs = {
        std::move(features), View(&length), std::move(cache_last_channel),
        std::move(cache_last_time), std::move(cache_last_channel_len)};

    auto out =
        encoder_sess_->Run({}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
                   encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());
    // out[0]: logit
    // out[1] logit_length
    // out[2:] states_next
    //
    // we need to remove out[1]

    std::vector<Ort::Value> ans;
    ans.reserve(out.size() - 1);

    for (int32_t i = 0; i != out.size(); ++i) {
      if (i == 1) {
        continue;
      }

      ans.push_back(std::move(out[i]));
    }

    return ans;
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> RunDecoder(
      Ort::Value targets, std::vector<Ort::Value> states) {
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Create the tensor with a single int32_t value of 1
    int32_t length_value = 1;
    std::vector<int64_t> length_shape = {1}; 

    Ort::Value targets_length = Ort::Value::CreateTensor<int32_t>(
        memory_info, &length_value, 1, length_shape.data(), length_shape.size()
    );
    
    std::vector<Ort::Value> decoder_inputs;
    decoder_inputs.reserve(2 + states.size());

    decoder_inputs.push_back(std::move(targets));
    decoder_inputs.push_back(std::move(targets_length));

    for (auto &s : states) {
      decoder_inputs.push_back(std::move(s));
    }

    auto decoder_out = decoder_sess_->Run(
        {}, decoder_input_names_ptr_.data(), decoder_inputs.data(),
        decoder_inputs.size(), decoder_output_names_ptr_.data(),
        decoder_output_names_ptr_.size());

    std::vector<Ort::Value> states_next;
    states_next.reserve(states.size());

    // decoder_out[0]: decoder_output
    // decoder_out[1]: decoder_output_length (discarded)
    // decoder_out[2:] states_next

    for (int32_t i = 0; i != states.size(); ++i) {
      states_next.push_back(std::move(decoder_out[i + 2]));
    }

    // we discard decoder_out[1]
    return {std::move(decoder_out[0]), std::move(states_next)};
  }

  Ort::Value RunJoiner(Ort::Value encoder_out, Ort::Value decoder_out) {
    std::array<Ort::Value, 2> joiner_input = {std::move(encoder_out),
                                              std::move(decoder_out)};
    auto logit =
        joiner_sess_->Run({}, joiner_input_names_ptr_.data(), joiner_input.data(),
                          joiner_input.size(), joiner_output_names_ptr_.data(),
                          joiner_output_names_ptr_.size());

    return std::move(logit[0]);
}

  std::vector<Ort::Value> GetDecoderInitStates(int32_t batch_size) const {
    std::array<int64_t, 3> s0_shape{pred_rnn_layers_, batch_size, pred_hidden_};
    Ort::Value s0 = Ort::Value::CreateTensor<float>(allocator_, s0_shape.data(),
                                                    s0_shape.size());

    Fill<float>(&s0, 0);

    std::array<int64_t, 3> s1_shape{pred_rnn_layers_, batch_size, pred_hidden_};

    Ort::Value s1 = Ort::Value::CreateTensor<float>(allocator_, s1_shape.data(),
                                                    s1_shape.size());

    Fill<float>(&s1, 0);

    std::vector<Ort::Value> states;

    states.reserve(2);
    states.push_back(std::move(s0));
    states.push_back(std::move(s1));

    return states;
  }

  int32_t ChunkSize() const { return window_size_; }

  int32_t ChunkShift() const { return chunk_shift_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }
  
  int32_t VocabSize() const { return vocab_size_; }

  OrtAllocator *Allocator() const { return allocator_; }

  std::string FeatureNormalizationMethod() const { return normalize_type_; }

  // Return a vector containing 3 tensors
  // - cache_last_channel
  // - cache_last_time_
  // - cache_last_channel_len
  std::vector<Ort::Value> GetInitStates() {
    std::vector<Ort::Value> ans;
    ans.reserve(3);
    ans.push_back(View(&cache_last_channel_));
    ans.push_back(View(&cache_last_time_));
    ans.push_back(View(&cache_last_channel_len_));

    return ans;
  }

private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

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
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");

    // need to increase by 1 since the blank token is not included in computing
    // vocab_size in NeMo.
    vocab_size_ += 1;

    SHERPA_ONNX_READ_META_DATA(window_size_, "window_size");
    SHERPA_ONNX_READ_META_DATA(chunk_shift_, "chunk_shift");
    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA_STR(normalize_type_, "normalize_type");
    SHERPA_ONNX_READ_META_DATA(pred_rnn_layers_, "pred_rnn_layers");
    SHERPA_ONNX_READ_META_DATA(pred_hidden_, "pred_hidden");

    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim1_,
                               "cache_last_channel_dim1");
    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim2_,
                               "cache_last_channel_dim2");
    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim3_,
                               "cache_last_channel_dim3");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim1_, "cache_last_time_dim1");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim2_, "cache_last_time_dim2");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim3_, "cache_last_time_dim3");

    if (normalize_type_ == "NA") {
      normalize_type_ = "";
    }

    InitStates();
  }
  
  void InitStates() {
    std::array<int64_t, 4> cache_last_channel_shape{1, cache_last_channel_dim1_,
                                                    cache_last_channel_dim2_,
                                                    cache_last_channel_dim3_};

    cache_last_channel_ = Ort::Value::CreateTensor<float>(
        allocator_, cache_last_channel_shape.data(),
        cache_last_channel_shape.size());

    Fill<float>(&cache_last_channel_, 0);

    std::array<int64_t, 4> cache_last_time_shape{
        1, cache_last_time_dim1_, cache_last_time_dim2_, cache_last_time_dim3_};

    cache_last_time_ = Ort::Value::CreateTensor<float>(
        allocator_, cache_last_time_shape.data(), cache_last_time_shape.size());

    Fill<float>(&cache_last_time_, 0);

    int64_t shape = 1;
    cache_last_channel_len_ =
        Ort::Value::CreateTensor<int64_t>(allocator_, &shape, 1);

    cache_last_channel_len_.GetTensorMutableData<int64_t>()[0] = 0;
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                  &decoder_output_names_ptr_);
  }

  void InitJoiner(void *model_data, size_t model_data_length) {
    joiner_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(joiner_sess_.get(), &joiner_input_names_,
                  &joiner_input_names_ptr_);

    GetOutputNames(joiner_sess_.get(), &joiner_output_names_,
                  &joiner_output_names_ptr_);
  }

 private:
  OnlineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
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

  int32_t window_size_;
  int32_t chunk_shift_;
  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 8;
  std::string normalize_type_;
  int32_t pred_rnn_layers_ = -1;
  int32_t pred_hidden_ = -1;

  int32_t cache_last_channel_dim1_;
  int32_t cache_last_channel_dim2_;
  int32_t cache_last_channel_dim3_;
  int32_t cache_last_time_dim1_;
  int32_t cache_last_time_dim2_;
  int32_t cache_last_time_dim3_;

  Ort::Value cache_last_channel_{nullptr};
  Ort::Value cache_last_time_{nullptr};
  Ort::Value cache_last_channel_len_{nullptr};
};

OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    AAssetManager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OnlineTransducerNeMoModel::~OnlineTransducerNeMoModel() = default;

std::vector<Ort::Value> 
OnlineTransducerNeMoModel::RunEncoder(Ort::Value features, 
                                      std::vector<Ort::Value> states) const {
    return impl_->RunEncoder(std::move(features), std::move(states));
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineTransducerNeMoModel::RunDecoder(Ort::Value targets,
                                      std::vector<Ort::Value> states) const {
  return impl_->RunDecoder(std::move(targets), std::move(states));
}

std::vector<Ort::Value> OnlineTransducerNeMoModel::GetDecoderInitStates(
    int32_t batch_size) const {
  return impl_->GetDecoderInitStates(batch_size);
}

Ort::Value OnlineTransducerNeMoModel::RunJoiner(Ort::Value encoder_out,
                                                Ort::Value decoder_out) const {
  return impl_->RunJoiner(std::move(encoder_out), std::move(decoder_out));
}


int32_t OnlineTransducerNeMoModel::ChunkSize() const { 
  return  impl_->ChunkSize();
  }

int32_t OnlineTransducerNeMoModel::ChunkShift() const { 
  return impl_->ChunkShift(); 
  }

int32_t OnlineTransducerNeMoModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

int32_t OnlineTransducerNeMoModel::VocabSize() const {
  return impl_->VocabSize();
}

OrtAllocator *OnlineTransducerNeMoModel::Allocator() const {
  return impl_->Allocator();
}

std::string OnlineTransducerNeMoModel::FeatureNormalizationMethod() const {
  return impl_->FeatureNormalizationMethod();
}

std::vector<Ort::Value> OnlineTransducerNeMoModel::GetInitStates() const {
  return impl_->GetInitStates();
}

}  // namespace sherpa_onnx