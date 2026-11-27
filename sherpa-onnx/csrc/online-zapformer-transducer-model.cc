// sherpa-onnx/csrc/online-zapformer-transducer-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zapformer-transducer-model.h"

#include <algorithm>
#include <array>
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
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

namespace {

// Return the layer index if name is prefix + "<int>", else -1.
int32_t LayerIndex(const std::string &name, const std::string &prefix) {
  if (name.compare(0, prefix.size(), prefix) != 0) return -1;
  std::string rest = name.substr(prefix.size());
  if (rest.empty() ||
      !std::all_of(rest.begin(), rest.end(),
                   [](char c) { return c >= '0' && c <= '9'; })) {
    return -1;
  }
  return std::stoi(rest);
}

}  // namespace

OnlineZapformerTransducerModel::OnlineZapformerTransducerModel(
    const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_ERROR),
      encoder_sess_opts_(GetSessionOptions(config)),
      decoder_sess_opts_(GetSessionOptions(config, "decoder")),
      joiner_sess_opts_(GetSessionOptions(config, "joiner")),
      config_(config),
      allocator_{} {
  encoder_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.encoder),
      encoder_sess_opts_);
  InitEncoder(nullptr, 0);

  decoder_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.decoder),
      decoder_sess_opts_);
  InitDecoder(nullptr, 0);

  joiner_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.joiner),
      joiner_sess_opts_);
  InitJoiner(nullptr, 0);
}

template <typename Manager>
OnlineZapformerTransducerModel::OnlineZapformerTransducerModel(
    Manager *mgr, const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_ERROR),
      encoder_sess_opts_(GetSessionOptions(config)),
      decoder_sess_opts_(GetSessionOptions(config, "decoder")),
      joiner_sess_opts_(GetSessionOptions(config, "joiner")),
      config_(config),
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

void OnlineZapformerTransducerModel::InitEncoder(void *model_data,
                                                 size_t model_data_length) {
  if (model_data) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, encoder_sess_opts_);
  } else if (!encoder_sess_) {
    SHERPA_ONNX_LOGE(
        "Please pass model data or initialize the encoder outside of "
        "this function");
    SHERPA_ONNX_EXIT(-1);
  }

  GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                &encoder_input_names_ptr_);

  GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                 &encoder_output_names_ptr_);

  Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
  if (config_.debug) {
    std::ostringstream os;
    os << "---encoder---\n";
    PrintModelMetadata(os, meta_data);
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
  SHERPA_ONNX_READ_META_DATA(T_, "T");
  SHERPA_ONNX_READ_META_DATA(decode_chunk_len_, "decode_chunk_len");

  std::string s;
  std::vector<float> norm_init_stats;
  std::vector<float> norm_init_len;
  SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(s, "norm_init_stats", "");
  if (!s.empty() && !SplitStringToFloats(s, ",", true, &norm_init_stats)) {
    SHERPA_ONNX_LOGE("Invalid norm_init_stats: %s", s.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(s, "norm_init_len", "");
  if (!s.empty() && !SplitStringToFloats(s, ",", true, &norm_init_len)) {
    SHERPA_ONNX_LOGE("Invalid norm_init_len: %s", s.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  // Encoder input 0 is x; inputs 1.. are the states. Encoder output 0 is
  // encoder_out; outputs 1.. are the new states, in the same order.
  size_t num_inputs = encoder_sess_->GetInputCount();
  if (num_inputs < 3 || encoder_sess_->GetOutputCount() != num_inputs) {
    SHERPA_ONNX_LOGE(
        "Unexpected zapformer encoder: %d inputs, %d outputs",
        static_cast<int32_t>(num_inputs),
        static_cast<int32_t>(encoder_sess_->GetOutputCount()));
    SHERPA_ONNX_EXIT(-1);
  }

  states_info_.clear();
  states_info_.reserve(num_inputs - 1);
  for (size_t i = 1; i != num_inputs; ++i) {
    // keep type_info alive: the shape info below is a view into it
    Ort::TypeInfo type_info = encoder_sess_->GetInputTypeInfo(i);
    auto info = type_info.GetTensorTypeAndShapeInfo();
    StateInfo st;
    st.name = encoder_input_names_[i];
    st.type = info.GetElementType();
    st.shape = info.GetShape();
    st.batch_axis = -1;
    for (int32_t k = 0; k != static_cast<int32_t>(st.shape.size()); ++k) {
      if (st.shape[k] < 0) {
        if (st.batch_axis != -1) {
          SHERPA_ONNX_LOGE("State %s has more than one dynamic axis",
                           st.name.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
        st.batch_axis = k;
        st.shape[k] = 1;
      }
    }
    if (st.batch_axis == -1) {
      std::ostringstream os;
      for (auto d : st.shape) os << d << " ";
      SHERPA_ONNX_LOGE("State %s has no dynamic (batch) axis. Shape: %s",
                       st.name.c_str(), os.str().c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    if (st.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        st.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      SHERPA_ONNX_LOGE("State %s: unsupported element type %d",
                       st.name.c_str(), static_cast<int32_t>(st.type));
      SHERPA_ONNX_EXIT(-1);
    }

    // The norm caches must start from the learned values in the metadata;
    // zero init gives wrong encoder output, so a missing entry is an error.
    st.init_value = 0;
    const std::vector<float> *init = nullptr;
    const char *key = nullptr;
    int32_t idx = LayerIndex(st.name, "cached_norm_stats_");
    if (idx >= 0) {
      init = &norm_init_stats;
      key = "norm_init_stats";
    } else if ((idx = LayerIndex(st.name, "cached_norm_len_")) >= 0) {
      init = &norm_init_len;
      key = "norm_init_len";
    }
    if (init) {
      if (idx >= static_cast<int32_t>(init->size())) {
        SHERPA_ONNX_LOGE(
            "Encoder metadata '%s' has no entry for state %s (%d entries). "
            "Please re-export the model with the latest "
            "zapformer/export-onnx-streaming.py from icefall.",
            key, st.name.c_str(), static_cast<int32_t>(init->size()));
        SHERPA_ONNX_EXIT(-1);
      }
      st.init_value = (*init)[idx];
    }
    states_info_.push_back(std::move(st));
  }

  if (config_.debug) {
    std::ostringstream os;
    os << "T: " << T_ << ", decode_chunk_len: " << decode_chunk_len_
       << ", num states: " << states_info_.size() << "\n";
    for (const auto &st : states_info_) {
      os << "  " << st.name << " [";
      for (auto d : st.shape) os << d << " ";
      os << "] batch_axis=" << st.batch_axis
         << (st.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? " int64"
                                                            : " float")
         << " init=" << st.init_value << "\n";
    }
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }
}

void OnlineZapformerTransducerModel::InitDecoder(void *model_data,
                                                 size_t model_data_length) {
  if (model_data) {
    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, decoder_sess_opts_);
  } else if (!decoder_sess_) {
    SHERPA_ONNX_LOGE(
        "Please pass model data or initialize the decoder outside of "
        "this function");
    SHERPA_ONNX_EXIT(-1);
  }

  GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                &decoder_input_names_ptr_);

  GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                 &decoder_output_names_ptr_);

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

void OnlineZapformerTransducerModel::InitJoiner(void *model_data,
                                                size_t model_data_length) {
  if (model_data) {
    joiner_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, joiner_sess_opts_);
  } else if (!joiner_sess_) {
    SHERPA_ONNX_LOGE(
        "Please pass model data or initialize the joiner outside of "
        "this function");
    SHERPA_ONNX_EXIT(-1);
  }

  GetInputNames(joiner_sess_.get(), &joiner_input_names_,
                &joiner_input_names_ptr_);

  GetOutputNames(joiner_sess_.get(), &joiner_output_names_,
                 &joiner_output_names_ptr_);

  if (config_.debug) {
    Ort::ModelMetadata meta_data = joiner_sess_->GetModelMetadata();
    std::ostringstream os;
    os << "---joiner---\n";
    PrintModelMetadata(os, meta_data);
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }
}

std::vector<Ort::Value> OnlineZapformerTransducerModel::StackStates(
    const std::vector<std::vector<Ort::Value>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());
  auto allocator =
      const_cast<OnlineZapformerTransducerModel *>(this)->allocator_;

  std::vector<const Ort::Value *> buf(batch_size);
  std::vector<Ort::Value> ans;
  ans.reserve(states_info_.size());

  for (size_t i = 0; i != states_info_.size(); ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][i];
    }
    const auto &st = states_info_[i];
    if (st.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      ans.push_back(Cat<int64_t>(allocator, buf, st.batch_axis));
    } else {
      ans.push_back(Cat(allocator, buf, st.batch_axis));
    }
  }
  return ans;
}

std::vector<std::vector<Ort::Value>>
OnlineZapformerTransducerModel::UnStackStates(
    const std::vector<Ort::Value> &states) const {
  SHERPA_ONNX_CHECK_EQ(states.size(), states_info_.size());
  auto allocator =
      const_cast<OnlineZapformerTransducerModel *>(this)->allocator_;

  int32_t batch_size = static_cast<int32_t>(
      states[0].GetTensorTypeAndShapeInfo()
          .GetShape()[states_info_[0].batch_axis]);

  std::vector<std::vector<Ort::Value>> ans(batch_size);
  for (auto &a : ans) a.reserve(states.size());

  for (size_t i = 0; i != states_info_.size(); ++i) {
    const auto &st = states_info_[i];
    std::vector<Ort::Value> v;
    if (st.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      v = Unbind<int64_t>(allocator, &states[i], st.batch_axis);
    } else {
      v = Unbind(allocator, &states[i], st.batch_axis);
    }
    SHERPA_ONNX_CHECK_EQ(static_cast<int32_t>(v.size()), batch_size);
    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }
  return ans;
}

std::vector<Ort::Value> OnlineZapformerTransducerModel::GetEncoderInitStates() {
  std::vector<Ort::Value> ans;
  ans.reserve(states_info_.size());
  for (const auto &st : states_info_) {
    if (st.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      auto v = Ort::Value::CreateTensor<int64_t>(allocator_, st.shape.data(),
                                                 st.shape.size());
      Fill<int64_t>(&v, static_cast<int64_t>(st.init_value));
      ans.push_back(std::move(v));
    } else {
      auto v = Ort::Value::CreateTensor<float>(allocator_, st.shape.data(),
                                               st.shape.size());
      Fill<float>(&v, st.init_value);
      ans.push_back(std::move(v));
    }
  }
  return ans;
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineZapformerTransducerModel::RunEncoder(Ort::Value features,
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
  next_states.reserve(encoder_out.size() - 1);
  for (size_t i = 1; i != encoder_out.size(); ++i) {
    next_states.push_back(std::move(encoder_out[i]));
  }
  return {std::move(encoder_out[0]), std::move(next_states)};
}

Ort::Value OnlineZapformerTransducerModel::RunDecoder(
    Ort::Value decoder_input) {
  auto decoder_out = decoder_sess_->Run(
      {}, decoder_input_names_ptr_.data(), &decoder_input, 1,
      decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());
  return std::move(decoder_out[0]);
}

Ort::Value OnlineZapformerTransducerModel::RunJoiner(Ort::Value encoder_out,
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
template OnlineZapformerTransducerModel::OnlineZapformerTransducerModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineZapformerTransducerModel::OnlineZapformerTransducerModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
