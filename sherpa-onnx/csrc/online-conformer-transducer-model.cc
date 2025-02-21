// sherpa-onnx/csrc/online-conformer-transducer-model.cc
//
// Copyright (c)  2023 Jingzhao Ou (jingzhao.ou@gmail.com)

#include "sherpa-onnx/csrc/online-conformer-transducer-model.h"

#include <algorithm>
#include <cassert>
#include <memory>
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

OnlineConformerTransducerModel::OnlineConformerTransducerModel(
    const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_ERROR),
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

template <typename Manager>
OnlineConformerTransducerModel::OnlineConformerTransducerModel(
    Manager *mgr, const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_ERROR),
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

void OnlineConformerTransducerModel::InitEncoder(void *model_data,
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
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
  }

  Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
  SHERPA_ONNX_READ_META_DATA(num_encoder_layers_, "num_encoder_layers");
  SHERPA_ONNX_READ_META_DATA(T_, "T");
  SHERPA_ONNX_READ_META_DATA(decode_chunk_len_, "decode_chunk_len");
  SHERPA_ONNX_READ_META_DATA(left_context_, "left_context");
  SHERPA_ONNX_READ_META_DATA(encoder_dim_, "encoder_dim");
  SHERPA_ONNX_READ_META_DATA(pad_length_, "pad_length");
  SHERPA_ONNX_READ_META_DATA(cnn_module_kernel_, "cnn_module_kernel");
}

void OnlineConformerTransducerModel::InitDecoder(void *model_data,
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
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
  }

  Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
  SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
  SHERPA_ONNX_READ_META_DATA(context_size_, "context_size");
}

void OnlineConformerTransducerModel::InitJoiner(void *model_data,
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

std::vector<Ort::Value> OnlineConformerTransducerModel::StackStates(
    const std::vector<std::vector<Ort::Value>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());

  std::vector<const Ort::Value *> attn_vec(batch_size);
  std::vector<const Ort::Value *> conv_vec(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    assert(states[i].size() == 2);
    attn_vec[i] = &states[i][0];
    conv_vec[i] = &states[i][1];
  }

  auto allocator =
      const_cast<OnlineConformerTransducerModel *>(this)->allocator_;

  Ort::Value attn = Cat(allocator, attn_vec, 2);
  Ort::Value conv = Cat(allocator, conv_vec, 2);

  std::vector<Ort::Value> ans;
  ans.reserve(2);
  ans.push_back(std::move(attn));
  ans.push_back(std::move(conv));

  return ans;
}

std::vector<std::vector<Ort::Value>>
OnlineConformerTransducerModel::UnStackStates(
    const std::vector<Ort::Value> &states) const {
  const int32_t batch_size =
      states[0].GetTensorTypeAndShapeInfo().GetShape()[2];
  assert(states.size() == 2);

  std::vector<std::vector<Ort::Value>> ans(batch_size);

  auto allocator =
      const_cast<OnlineConformerTransducerModel *>(this)->allocator_;

  std::vector<Ort::Value> attn_vec = Unbind(allocator, &states[0], 2);
  std::vector<Ort::Value> conv_vec = Unbind(allocator, &states[1], 2);

  assert(attn_vec.size() == batch_size);
  assert(conv_vec.size() == batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    ans[i].push_back(std::move(attn_vec[i]));
    ans[i].push_back(std::move(conv_vec[i]));
  }

  return ans;
}

std::vector<Ort::Value> OnlineConformerTransducerModel::GetEncoderInitStates() {
  // Please see
  // https://github.com/k2-fsa/icefall/blob/86b0db6eb9c84d9bc90a71d92774fe2a7f73e6ab/egs/librispeech/ASR/pruned_transducer_stateless5/conformer.py#L203
  // for details
  constexpr int32_t kBatchSize = 1;
  std::array<int64_t, 4> h_shape{num_encoder_layers_, left_context_, kBatchSize,
                                 encoder_dim_};
  Ort::Value h = Ort::Value::CreateTensor<float>(allocator_, h_shape.data(),
                                                 h_shape.size());

  Fill<float>(&h, 0);

  std::array<int64_t, 4> c_shape{num_encoder_layers_, cnn_module_kernel_ - 1,
                                 kBatchSize, encoder_dim_};

  Ort::Value c = Ort::Value::CreateTensor<float>(allocator_, c_shape.data(),
                                                 c_shape.size());

  Fill<float>(&c, 0);

  std::vector<Ort::Value> states;

  states.reserve(2);
  states.push_back(std::move(h));
  states.push_back(std::move(c));

  return states;
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineConformerTransducerModel::RunEncoder(Ort::Value features,
                                           std::vector<Ort::Value> states,
                                           Ort::Value processed_frames) {
  std::array<Ort::Value, 4> encoder_inputs = {
      std::move(features), std::move(states[0]), std::move(states[1]),
      std::move(processed_frames)};

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

Ort::Value OnlineConformerTransducerModel::RunDecoder(
    Ort::Value decoder_input) {
  auto decoder_out = decoder_sess_->Run(
      {}, decoder_input_names_ptr_.data(), &decoder_input, 1,
      decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());
  return std::move(decoder_out[0]);
}

Ort::Value OnlineConformerTransducerModel::RunJoiner(Ort::Value encoder_out,
                                                     Ort::Value decoder_out) {
  std::array<Ort::Value, 2> joiner_input = {std::move(encoder_out),
                                            std::move(decoder_out)};
  auto logit =
      joiner_sess_->Run({}, joiner_input_names_ptr_.data(), joiner_input.data(),
                        joiner_input.size(), joiner_output_names_ptr_.data(),
                        joiner_output_names_ptr_.size());

  return std::move(logit[0]);
}

#if __ANDROID_API__ >= 9
template OnlineConformerTransducerModel::OnlineConformerTransducerModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineConformerTransducerModel::OnlineConformerTransducerModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
