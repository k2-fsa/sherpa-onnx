// sherpa-onnx/csrc/online-zipformer2-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zipformer2-transducer-model.h"

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

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

OnlineZipformer2TransducerModel::OnlineZipformer2TransducerModel(
    const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_WARNING),
      config_(config),
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
OnlineZipformer2TransducerModel::OnlineZipformer2TransducerModel(
    AAssetManager *mgr, const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_WARNING),
      config_(config),
      sess_opts_(GetSessionOptions(config)),
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
#endif

void OnlineZipformer2TransducerModel::InitEncoder(void *model_data,
                                                  size_t model_data_length) {
  encoder_sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                                 model_data_length, sess_opts_);

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
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
  SHERPA_ONNX_READ_META_DATA_VEC(encoder_dims_, "encoder_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(query_head_dims_, "query_head_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(value_head_dims_, "value_head_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(num_heads_, "num_heads");
  SHERPA_ONNX_READ_META_DATA_VEC(num_encoder_layers_, "num_encoder_layers");
  SHERPA_ONNX_READ_META_DATA_VEC(cnn_module_kernels_, "cnn_module_kernels");
  SHERPA_ONNX_READ_META_DATA_VEC(left_context_len_, "left_context_len");

  SHERPA_ONNX_READ_META_DATA(T_, "T");
  SHERPA_ONNX_READ_META_DATA(decode_chunk_len_, "decode_chunk_len");

  if (config_.debug) {
    auto print = [](const std::vector<int32_t> &v, const char *name) {
      fprintf(stderr, "%s: ", name);
      for (auto i : v) {
        fprintf(stderr, "%d ", i);
      }
      fprintf(stderr, "\n");
    };
    print(encoder_dims_, "encoder_dims");
    print(query_head_dims_, "query_head_dims");
    print(value_head_dims_, "value_head_dims");
    print(num_heads_, "num_heads");
    print(num_encoder_layers_, "num_encoder_layers");
    print(cnn_module_kernels_, "cnn_module_kernels");
    print(left_context_len_, "left_context_len");
    SHERPA_ONNX_LOGE("T: %d", T_);
    SHERPA_ONNX_LOGE("decode_chunk_len_: %d", decode_chunk_len_);
  }
}

void OnlineZipformer2TransducerModel::InitDecoder(void *model_data,
                                                  size_t model_data_length) {
  decoder_sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                                 model_data_length, sess_opts_);

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

void OnlineZipformer2TransducerModel::InitJoiner(void *model_data,
                                                 size_t model_data_length) {
  joiner_sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                                model_data_length, sess_opts_);

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

std::vector<Ort::Value> OnlineZipformer2TransducerModel::StackStates(
    const std::vector<std::vector<Ort::Value>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());
  int32_t num_encoders = static_cast<int32_t>(num_encoder_layers_.size());

  std::vector<const Ort::Value *> buf(batch_size);

  std::vector<Ort::Value> ans;
  int32_t num_states = static_cast<int32_t>(states[0].size());
  ans.reserve(num_states);

  for (int32_t i = 0; i != (num_states - 2) / 6; ++i) {
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][6 * i];
      }
      auto v = Cat(allocator_, buf, 1);
      ans.push_back(std::move(v));
    }
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][6 * i + 1];
      }
      auto v = Cat(allocator_, buf, 1);
      ans.push_back(std::move(v));
    }
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][6 * i + 2];
      }
      auto v = Cat(allocator_, buf, 1);
      ans.push_back(std::move(v));
    }
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][6 * i + 3];
      }
      auto v = Cat(allocator_, buf, 1);
      ans.push_back(std::move(v));
    }
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][6 * i + 4];
      }
      auto v = Cat(allocator_, buf, 0);
      ans.push_back(std::move(v));
    }
    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][6 * i + 5];
      }
      auto v = Cat(allocator_, buf, 0);
      ans.push_back(std::move(v));
    }
  }

  {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_states - 2];
    }
    auto v = Cat(allocator_, buf, 0);
    ans.push_back(std::move(v));
  }

  {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_states - 1];
    }
    auto v = Cat<int64_t>(allocator_, buf, 0);
    ans.push_back(std::move(v));
  }
  return ans;
}

std::vector<std::vector<Ort::Value>>
OnlineZipformer2TransducerModel::UnStackStates(
    const std::vector<Ort::Value> &states) const {
  int32_t m = std::accumulate(num_encoder_layers_.begin(),
                              num_encoder_layers_.end(), 0);
  assert(states.size() == m * 6 + 2);

  int32_t batch_size = states[0].GetTensorTypeAndShapeInfo().GetShape()[1];
  int32_t num_encoders = num_encoder_layers_.size();

  std::vector<std::vector<Ort::Value>> ans;
  ans.resize(batch_size);

  for (int32_t i = 0; i != m; ++i) {
    {
      auto v = Unbind(allocator_, &states[i * 6], 1);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind(allocator_, &states[i * 6 + 1], 1);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind(allocator_, &states[i * 6 + 2], 1);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind(allocator_, &states[i * 6 + 3], 1);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind(allocator_, &states[i * 6 + 4], 0);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind(allocator_, &states[i * 6 + 5], 0);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
  }

  {
    auto v = Unbind(allocator_, &states[m * 6], 0);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }
  {
    auto v = Unbind<int64_t>(allocator_, &states[m * 6 + 1], 0);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  return ans;
}

std::vector<Ort::Value>
OnlineZipformer2TransducerModel::GetEncoderInitStates() {
  std::vector<Ort::Value> ans;
  int32_t n = static_cast<int32_t>(encoder_dims_.size());
  int32_t m = std::accumulate(num_encoder_layers_.begin(),
                              num_encoder_layers_.end(), 0);
  ans.reserve(m * 6 + 2);

  for (int32_t i = 0; i != n; ++i) {
    int32_t num_layers = num_encoder_layers_[i];
    int32_t key_dim = query_head_dims_[i] * num_heads_[i];
    int32_t value_dim = value_head_dims_[i] * num_heads_[i];
    int32_t nonlin_attn_head_dim = 3 * encoder_dims_[i] / 4;

    for (int32_t j = 0; j != num_layers; ++j) {
      {
        std::array<int64_t, 3> s{left_context_len_[i], 1, key_dim};
        auto v =
            Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
        Fill(&v, 0);
        ans.push_back(std::move(v));
      }

      {
        std::array<int64_t, 4> s{1, 1, left_context_len_[i],
                                 nonlin_attn_head_dim};
        auto v =
            Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
        Fill(&v, 0);
        ans.push_back(std::move(v));
      }

      {
        std::array<int64_t, 3> s{left_context_len_[i], 1, value_dim};
        auto v =
            Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
        Fill(&v, 0);
        ans.push_back(std::move(v));
      }

      {
        std::array<int64_t, 3> s{left_context_len_[i], 1, value_dim};
        auto v =
            Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
        Fill(&v, 0);
        ans.push_back(std::move(v));
      }

      {
        std::array<int64_t, 3> s{1, encoder_dims_[i],
                                 cnn_module_kernels_[i] / 2};
        auto v =
            Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
        Fill(&v, 0);
        ans.push_back(std::move(v));
      }

      {
        std::array<int64_t, 3> s{1, encoder_dims_[i],
                                 cnn_module_kernels_[i] / 2};
        auto v =
            Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
        Fill(&v, 0);
        ans.push_back(std::move(v));
      }
    }
  }

  {
    std::array<int64_t, 4> s{1, 128, 3, 19};
    auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
    Fill(&v, 0);
    ans.push_back(std::move(v));
  }

  {
    std::array<int64_t, 1> s{1};
    auto v = Ort::Value::CreateTensor<int64_t>(allocator_, s.data(), s.size());
    Fill<int64_t>(&v, 0);
    ans.push_back(std::move(v));
  }
  return ans;
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineZipformer2TransducerModel::RunEncoder(Ort::Value features,
                                            std::vector<Ort::Value> states,
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

Ort::Value OnlineZipformer2TransducerModel::RunDecoder(
    Ort::Value decoder_input) {
  auto decoder_out = decoder_sess_->Run(
      {}, decoder_input_names_ptr_.data(), &decoder_input, 1,
      decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());
  return std::move(decoder_out[0]);
}

Ort::Value OnlineZipformer2TransducerModel::RunJoiner(Ort::Value encoder_out,
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
