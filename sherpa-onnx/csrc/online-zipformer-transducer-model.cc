// sherpa-onnx/csrc/online-zipformer-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zipformer-transducer-model.h"

#include <assert.h>

#include <algorithm>
#include <memory>
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
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

OnlineZipformerTransducerModel::OnlineZipformerTransducerModel(
    const OnlineTransducerModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_WARNING),
      config_(config),
      sess_opts_{},
      allocator_{} {
  sess_opts_.SetIntraOpNumThreads(config.num_threads);
  sess_opts_.SetInterOpNumThreads(config.num_threads);

  {
    auto buf = ReadFile(config.encoder_filename);
    InitEncoder(buf.data(), buf.size());
  }

  {
    auto buf = ReadFile(config.decoder_filename);
    InitDecoder(buf.data(), buf.size());
  }

  {
    auto buf = ReadFile(config.joiner_filename);
    InitJoiner(buf.data(), buf.size());
  }
}

#if __ANDROID_API__ >= 9
OnlineZipformerTransducerModel::OnlineZipformerTransducerModel(
    AAssetManager *mgr, const OnlineTransducerModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_WARNING),
      config_(config),
      sess_opts_{},
      allocator_{} {
  sess_opts_.SetIntraOpNumThreads(config.num_threads);
  sess_opts_.SetInterOpNumThreads(config.num_threads);

  {
    auto buf = ReadFile(mgr, config.encoder_filename);
    InitEncoder(buf.data(), buf.size());
  }

  {
    auto buf = ReadFile(mgr, config.decoder_filename);
    InitDecoder(buf.data(), buf.size());
  }

  {
    auto buf = ReadFile(mgr, config.joiner_filename);
    InitJoiner(buf.data(), buf.size());
  }
}
#endif

void OnlineZipformerTransducerModel::InitEncoder(void *model_data,
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
    fprintf(stderr, "%s\n", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
  SHERPA_ONNX_READ_META_DATA_VEC(encoder_dims_, "encoder_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(attention_dims_, "attention_dims");
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
    print(attention_dims_, "attention_dims");
    print(num_encoder_layers_, "num_encoder_layers");
    print(cnn_module_kernels_, "cnn_module_kernels");
    print(left_context_len_, "left_context_len");
    fprintf(stderr, "T: %d\n", T_);
    fprintf(stderr, "decode_chunk_len_: %d\n", decode_chunk_len_);
  }
}

void OnlineZipformerTransducerModel::InitDecoder(void *model_data,
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
    fprintf(stderr, "%s\n", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
  SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
  SHERPA_ONNX_READ_META_DATA(context_size_, "context_size");
}

void OnlineZipformerTransducerModel::InitJoiner(void *model_data,
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
    fprintf(stderr, "%s\n", os.str().c_str());
  }
}

std::vector<Ort::Value> OnlineZipformerTransducerModel::StackStates(
    const std::vector<std::vector<Ort::Value>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());
  int32_t num_encoders = static_cast<int32_t>(num_encoder_layers_.size());

  std::vector<const Ort::Value *> buf(batch_size);

  std::vector<Ort::Value> ans;
  ans.reserve(states[0].size());

  // cached_len
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][i];
    }
    auto v = Cat<int64_t>(allocator_, buf, 1);  // (num_layers, 1)
    ans.push_back(std::move(v));
  }

  // cached_avg
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders + i];
    }
    auto v = Cat(allocator_, buf, 1);  // (num_layers, 1, encoder_dims)
    ans.push_back(std::move(v));
  }

  // cached_key
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 2 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_val
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 3 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims/2)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_val2
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 4 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims/2)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_conv1
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 5 + i];
    }
    // (num_layers, 1, encoder_dims, cnn_module_kernels-1)
    auto v = Cat(allocator_, buf, 1);
    ans.push_back(std::move(v));
  }

  // cached_conv2
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 6 + i];
    }
    // (num_layers, 1, encoder_dims, cnn_module_kernels-1)
    auto v = Cat(allocator_, buf, 1);
    ans.push_back(std::move(v));
  }

  return ans;
}

std::vector<std::vector<Ort::Value>>
OnlineZipformerTransducerModel::UnStackStates(
    const std::vector<Ort::Value> &states) const {
  assert(states.size() == num_encoder_layers_.size() * 7);

  int32_t batch_size = states[0].GetTensorTypeAndShapeInfo().GetShape()[1];
  int32_t num_encoders = num_encoder_layers_.size();

  std::vector<std::vector<Ort::Value>> ans;
  ans.resize(batch_size);

  // cached_len
  for (int32_t i = 0; i != num_encoders; ++i) {
    auto v = Unbind<int64_t>(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_avg
  for (int32_t i = num_encoders; i != 2 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_key
  for (int32_t i = 2 * num_encoders; i != 3 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_val
  for (int32_t i = 3 * num_encoders; i != 4 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_val2
  for (int32_t i = 4 * num_encoders; i != 5 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_conv1
  for (int32_t i = 5 * num_encoders; i != 6 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_conv2
  for (int32_t i = 6 * num_encoders; i != 7 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  return ans;
}

std::vector<Ort::Value> OnlineZipformerTransducerModel::GetEncoderInitStates() {
  // Please see
  // https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py#L673
  // for details

  int32_t n = static_cast<int32_t>(encoder_dims_.size());
  std::vector<Ort::Value> cached_len_vec;
  std::vector<Ort::Value> cached_avg_vec;
  std::vector<Ort::Value> cached_key_vec;
  std::vector<Ort::Value> cached_val_vec;
  std::vector<Ort::Value> cached_val2_vec;
  std::vector<Ort::Value> cached_conv1_vec;
  std::vector<Ort::Value> cached_conv2_vec;

  cached_len_vec.reserve(n);
  cached_avg_vec.reserve(n);
  cached_key_vec.reserve(n);
  cached_val_vec.reserve(n);
  cached_val2_vec.reserve(n);
  cached_conv1_vec.reserve(n);
  cached_conv2_vec.reserve(n);

  for (int32_t i = 0; i != n; ++i) {
    {
      std::array<int64_t, 2> s{num_encoder_layers_[i], 1};
      auto v =
          Ort::Value::CreateTensor<int64_t>(allocator_, s.data(), s.size());
      Fill<int64_t>(&v, 0);
      cached_len_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 3> s{num_encoder_layers_[i], 1, encoder_dims_[i]};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_avg_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i]};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_key_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i] / 2};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_val_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i] / 2};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_val2_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], 1, encoder_dims_[i],
                               cnn_module_kernels_[i] - 1};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_conv1_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], 1, encoder_dims_[i],
                               cnn_module_kernels_[i] - 1};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_conv2_vec.push_back(std::move(v));
    }
  }

  std::vector<Ort::Value> ans;
  ans.reserve(n * 7);

  for (auto &v : cached_len_vec) ans.push_back(std::move(v));
  for (auto &v : cached_avg_vec) ans.push_back(std::move(v));
  for (auto &v : cached_key_vec) ans.push_back(std::move(v));
  for (auto &v : cached_val_vec) ans.push_back(std::move(v));
  for (auto &v : cached_val2_vec) ans.push_back(std::move(v));
  for (auto &v : cached_conv1_vec) ans.push_back(std::move(v));
  for (auto &v : cached_conv2_vec) ans.push_back(std::move(v));

  return ans;
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineZipformerTransducerModel::RunEncoder(Ort::Value features,
                                           std::vector<Ort::Value> states) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

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

Ort::Value OnlineZipformerTransducerModel::RunDecoder(
    Ort::Value decoder_input) {
  auto decoder_out = decoder_sess_->Run(
      {}, decoder_input_names_ptr_.data(), &decoder_input, 1,
      decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());
  return std::move(decoder_out[0]);
}

Ort::Value OnlineZipformerTransducerModel::RunJoiner(Ort::Value encoder_out,
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
