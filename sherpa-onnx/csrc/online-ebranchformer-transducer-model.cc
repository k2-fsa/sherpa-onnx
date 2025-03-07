// sherpa-onnx/csrc/online-ebranchformer-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
//                2025  Brno University of Technology (author: Karel Vesely)

#include "sherpa-onnx/csrc/online-ebranchformer-transducer-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
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

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

OnlineEbranchformerTransducerModel::OnlineEbranchformerTransducerModel(
    const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_ERROR),
      encoder_sess_opts_(GetSessionOptions(config)),
      decoder_sess_opts_(GetSessionOptions(config, "decoder")),
      joiner_sess_opts_(GetSessionOptions(config, "joiner")),
      config_(config),
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

template <typename Manager>
OnlineEbranchformerTransducerModel::OnlineEbranchformerTransducerModel(
    Manager *mgr, const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_ERROR),
      config_(config),
      encoder_sess_opts_(GetSessionOptions(config)),
      decoder_sess_opts_(GetSessionOptions(config)),
      joiner_sess_opts_(GetSessionOptions(config)),
      allocator_{} {
  {
    auto buf = ReadFile(mgr, config.transducer.encoder);
    InitEncoder(buf.data(), buf.size());
  }

  {
    auto buf = ReadFile(mgr, config.transducer.decoder);
    InitDecoder(buf.data(), buf.size());
  }

  {
    auto buf = ReadFile(mgr, config.transducer.joiner);
    InitJoiner(buf.data(), buf.size());
  }
}

void OnlineEbranchformerTransducerModel::InitEncoder(void *model_data,
                                                     size_t model_data_length) {
  encoder_sess_ = std::make_unique<Ort::Session>(
      env_, model_data, model_data_length, encoder_sess_opts_);

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
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
  }

  Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

  SHERPA_ONNX_READ_META_DATA(decode_chunk_len_, "decode_chunk_len");
  SHERPA_ONNX_READ_META_DATA(T_, "T");

  SHERPA_ONNX_READ_META_DATA(num_hidden_layers_, "num_hidden_layers");
  SHERPA_ONNX_READ_META_DATA(hidden_size_, "hidden_size");
  SHERPA_ONNX_READ_META_DATA(intermediate_size_, "intermediate_size");
  SHERPA_ONNX_READ_META_DATA(csgu_kernel_size_, "csgu_kernel_size");
  SHERPA_ONNX_READ_META_DATA(merge_conv_kernel_, "merge_conv_kernel");
  SHERPA_ONNX_READ_META_DATA(left_context_len_, "left_context_len");
  SHERPA_ONNX_READ_META_DATA(num_heads_, "num_heads");
  SHERPA_ONNX_READ_META_DATA(head_dim_, "head_dim");

  if (config_.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("T: %{public}d", T_);
    SHERPA_ONNX_LOGE("decode_chunk_len_: %{public}d", decode_chunk_len_);

    SHERPA_ONNX_LOGE("num_hidden_layers_: %{public}d", num_hidden_layers_);
    SHERPA_ONNX_LOGE("hidden_size_: %{public}d", hidden_size_);
    SHERPA_ONNX_LOGE("intermediate_size_: %{public}d", intermediate_size_);
    SHERPA_ONNX_LOGE("csgu_kernel_size_: %{public}d", csgu_kernel_size_);
    SHERPA_ONNX_LOGE("merge_conv_kernel_: %{public}d", merge_conv_kernel_);
    SHERPA_ONNX_LOGE("left_context_len_: %{public}d", left_context_len_);
    SHERPA_ONNX_LOGE("num_heads_: %{public}d", num_heads_);
    SHERPA_ONNX_LOGE("head_dim_: %{public}d", head_dim_);
#else
    SHERPA_ONNX_LOGE("T: %d", T_);
    SHERPA_ONNX_LOGE("decode_chunk_len_: %d", decode_chunk_len_);

    SHERPA_ONNX_LOGE("num_hidden_layers_: %d", num_hidden_layers_);
    SHERPA_ONNX_LOGE("hidden_size_: %d", hidden_size_);
    SHERPA_ONNX_LOGE("intermediate_size_: %d", intermediate_size_);
    SHERPA_ONNX_LOGE("csgu_kernel_size_: %d", csgu_kernel_size_);
    SHERPA_ONNX_LOGE("merge_conv_kernel_: %d", merge_conv_kernel_);
    SHERPA_ONNX_LOGE("left_context_len_: %d", left_context_len_);
    SHERPA_ONNX_LOGE("num_heads_: %d", num_heads_);
    SHERPA_ONNX_LOGE("head_dim_: %d", head_dim_);
#endif
  }
}

void OnlineEbranchformerTransducerModel::InitDecoder(void *model_data,
                                                     size_t model_data_length) {
  decoder_sess_ = std::make_unique<Ort::Session>(
      env_, model_data, model_data_length, decoder_sess_opts_);

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
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
  SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
  SHERPA_ONNX_READ_META_DATA(context_size_, "context_size");
}

void OnlineEbranchformerTransducerModel::InitJoiner(void *model_data,
                                                    size_t model_data_length) {
  joiner_sess_ = std::make_unique<Ort::Session>(
      env_, model_data, model_data_length, joiner_sess_opts_);

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
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }
}

std::vector<Ort::Value> OnlineEbranchformerTransducerModel::StackStates(
    const std::vector<std::vector<Ort::Value>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());

  std::vector<const Ort::Value *> buf(batch_size);

  auto allocator =
      const_cast<OnlineEbranchformerTransducerModel *>(this)->allocator_;

  std::vector<Ort::Value> ans;
  int32_t num_states = static_cast<int32_t>(states[0].size());
  ans.reserve(num_states);

  for (int32_t i = 0; i != num_hidden_layers_; ++i) {
    {  // cached_key
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][4 * i];
      }
      auto v = Cat(allocator, buf, /* axis */ 0);
      ans.push_back(std::move(v));
    }
    {  // cached_value
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][4 * i + 1];
      }
      auto v = Cat(allocator, buf, 0);
      ans.push_back(std::move(v));
    }
    {  // cached_conv
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][4 * i + 2];
      }
      auto v = Cat(allocator, buf, 0);
      ans.push_back(std::move(v));
    }
    {  // cached_conv_fusion
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][4 * i + 3];
      }
      auto v = Cat(allocator, buf, 0);
      ans.push_back(std::move(v));
    }
  }

  {  // processed_lens
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_states - 1];
    }
    auto v = Cat<int64_t>(allocator, buf, 0);
    ans.push_back(std::move(v));
  }

  return ans;
}

std::vector<std::vector<Ort::Value>>
OnlineEbranchformerTransducerModel::UnStackStates(
    const std::vector<Ort::Value> &states) const {
  assert(static_cast<int32_t>(states.size()) == num_hidden_layers_ * 4 + 1);

  int32_t batch_size = states[0].GetTensorTypeAndShapeInfo().GetShape()[0];

  auto allocator =
      const_cast<OnlineEbranchformerTransducerModel *>(this)->allocator_;

  std::vector<std::vector<Ort::Value>> ans;
  ans.resize(batch_size);

  for (int32_t i = 0; i != num_hidden_layers_; ++i) {
    {  // cached_key
      auto v = Unbind(allocator, &states[i * 4], /* axis */ 0);
      assert(static_cast<int32_t>(v.size()) == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {  // cached_value
      auto v = Unbind(allocator, &states[i * 4 + 1], 0);
      assert(static_cast<int32_t>(v.size()) == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {  // cached_conv
      auto v = Unbind(allocator, &states[i * 4 + 2], 0);
      assert(static_cast<int32_t>(v.size()) == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {  // cached_conv_fusion
      auto v = Unbind(allocator, &states[i * 4 + 3], 0);
      assert(static_cast<int32_t>(v.size()) == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
  }

  {  // processed_lens
    auto v = Unbind<int64_t>(allocator, &states.back(), 0);
    assert(static_cast<int32_t>(v.size()) == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  return ans;
}

std::vector<Ort::Value>
OnlineEbranchformerTransducerModel::GetEncoderInitStates() {
  std::vector<Ort::Value> ans;

  ans.reserve(num_hidden_layers_ * 4 + 1);

  int32_t left_context_conv = csgu_kernel_size_ - 1;
  int32_t channels_conv = intermediate_size_ / 2;

  int32_t left_context_conv_fusion = merge_conv_kernel_ - 1;
  int32_t channels_conv_fusion = 2 * hidden_size_;

  for (int32_t i = 0; i != num_hidden_layers_; ++i) {
    {  // cached_key_{i}
      std::array<int64_t, 4> s{1, num_heads_, left_context_len_, head_dim_};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      ans.push_back(std::move(v));
    }

    {  // cahced_value_{i}
      std::array<int64_t, 4> s{1, num_heads_, left_context_len_, head_dim_};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      ans.push_back(std::move(v));
    }

    {  // cached_conv_{i}
      std::array<int64_t, 3> s{1, channels_conv, left_context_conv};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      ans.push_back(std::move(v));
    }

    {  // cached_conv_fusion_{i}
      std::array<int64_t, 3> s{1, channels_conv_fusion,
                               left_context_conv_fusion};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      ans.push_back(std::move(v));
    }
  }  // num_hidden_layers_

  {  // processed_lens
    std::array<int64_t, 1> s{1};
    auto v = Ort::Value::CreateTensor<int64_t>(allocator_, s.data(), s.size());
    Fill<int64_t>(&v, 0);
    ans.push_back(std::move(v));
  }

  return ans;
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineEbranchformerTransducerModel::RunEncoder(
    Ort::Value features, std::vector<Ort::Value> states,
    Ort::Value /* processed_frames */) {
  std::vector<Ort::Value> encoder_inputs;
  encoder_inputs.reserve(1 + states.size());

  encoder_inputs.push_back(std::move(features));
  for (auto &v : states) {
    encoder_inputs.push_back(std::move(v));
  }

  auto encoder_out = encoder_sess_->Run(
      {}, encoder_input_names_ptr_.data(), encoder_inputs.data(),
      encoder_inputs.size(), encoder_output_names_ptr_.data(),
      encoder_output_names_ptr_.size());

  std::vector<Ort::Value> next_states;
  next_states.reserve(states.size());

  for (int32_t i = 1; i != static_cast<int32_t>(encoder_out.size()); ++i) {
    next_states.push_back(std::move(encoder_out[i]));
  }
  return {std::move(encoder_out[0]), std::move(next_states)};
}

Ort::Value OnlineEbranchformerTransducerModel::RunDecoder(
    Ort::Value decoder_input) {
  auto decoder_out = decoder_sess_->Run(
      {}, decoder_input_names_ptr_.data(), &decoder_input, 1,
      decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());
  return std::move(decoder_out[0]);
}

Ort::Value OnlineEbranchformerTransducerModel::RunJoiner(
    Ort::Value encoder_out, Ort::Value decoder_out) {
  std::array<Ort::Value, 2> joiner_input = {std::move(encoder_out),
                                            std::move(decoder_out)};
  auto logit =
      joiner_sess_->Run({}, joiner_input_names_ptr_.data(), joiner_input.data(),
                        joiner_input.size(), joiner_output_names_ptr_.data(),
                        joiner_output_names_ptr_.size());

  return std::move(logit[0]);
}

#if __ANDROID_API__ >= 9
template OnlineEbranchformerTransducerModel::OnlineEbranchformerTransducerModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineEbranchformerTransducerModel::OnlineEbranchformerTransducerModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
