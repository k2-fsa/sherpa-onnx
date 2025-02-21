// sherpa-onnx/csrc/offline-fire-red-asr-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-fire-red-asr-model.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineFireRedAsrModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.fire_red_asr.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.fire_red_asr.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.fire_red_asr.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.fire_red_asr.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  std::pair<Ort::Value, Ort::Value> ForwardEncoder(Ort::Value features,
                                                   Ort::Value features_length) {
    std::array<Ort::Value, 2> inputs{std::move(features),
                                     std::move(features_length)};

    auto encoder_out = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());

    return {std::move(encoder_out[0]), std::move(encoder_out[1])};
  }

  std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
             Ort::Value>
  ForwardDecoder(Ort::Value tokens, Ort::Value n_layer_self_k_cache,
                 Ort::Value n_layer_self_v_cache, Ort::Value n_layer_cross_k,
                 Ort::Value n_layer_cross_v, Ort::Value offset) {
    std::array<Ort::Value, 6> decoder_input = {std::move(tokens),
                                               std::move(n_layer_self_k_cache),
                                               std::move(n_layer_self_v_cache),
                                               std::move(n_layer_cross_k),
                                               std::move(n_layer_cross_v),
                                               std::move(offset)};

    auto decoder_out = decoder_sess_->Run(
        {}, decoder_input_names_ptr_.data(), decoder_input.data(),
        decoder_input.size(), decoder_output_names_ptr_.data(),
        decoder_output_names_ptr_.size());

    return std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value,
                      Ort::Value, Ort::Value>{
        std::move(decoder_out[0]),   std::move(decoder_out[1]),
        std::move(decoder_out[2]),   std::move(decoder_input[3]),
        std::move(decoder_input[4]), std::move(decoder_input[5])};
  }

  std::pair<Ort::Value, Ort::Value> GetInitialSelfKVCache() {
    int32_t batch_size = 1;
    std::array<int64_t, 5> shape{meta_data_.num_decoder_layers, batch_size,
                                 meta_data_.max_len, meta_data_.num_head,
                                 meta_data_.head_dim};

    Ort::Value n_layer_self_k_cache = Ort::Value::CreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    Ort::Value n_layer_self_v_cache = Ort::Value::CreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    auto n = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];

    float *p_k = n_layer_self_k_cache.GetTensorMutableData<float>();
    float *p_v = n_layer_self_v_cache.GetTensorMutableData<float>();

    memset(p_k, 0, sizeof(float) * n);
    memset(p_v, 0, sizeof(float) * n);

    return {std::move(n_layer_self_k_cache), std::move(n_layer_self_v_cache)};
  }

  OrtAllocator *Allocator() { return allocator_; }

  const OfflineFireRedAsrModelMetaData &GetModelMetadata() const {
    return meta_data_;
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
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_decoder_layers,
                               "num_decoder_layers");
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_head, "num_head");
    SHERPA_ONNX_READ_META_DATA(meta_data_.head_dim, "head_dim");
    SHERPA_ONNX_READ_META_DATA(meta_data_.sos_id, "sos");
    SHERPA_ONNX_READ_META_DATA(meta_data_.eos_id, "eos");
    SHERPA_ONNX_READ_META_DATA(meta_data_.max_len, "max_len");

    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.mean, "cmvn_mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.inv_stddev,
                                         "cmvn_inv_stddev");
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);
  }

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  OfflineFireRedAsrModelMetaData meta_data_;
};

OfflineFireRedAsrModel::OfflineFireRedAsrModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineFireRedAsrModel::OfflineFireRedAsrModel(Manager *mgr,
                                               const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineFireRedAsrModel::~OfflineFireRedAsrModel() = default;

std::pair<Ort::Value, Ort::Value> OfflineFireRedAsrModel::ForwardEncoder(
    Ort::Value features, Ort::Value features_length) const {
  return impl_->ForwardEncoder(std::move(features), std::move(features_length));
}

std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
           Ort::Value>
OfflineFireRedAsrModel::ForwardDecoder(Ort::Value tokens,
                                       Ort::Value n_layer_self_k_cache,
                                       Ort::Value n_layer_self_v_cache,
                                       Ort::Value n_layer_cross_k,
                                       Ort::Value n_layer_cross_v,
                                       Ort::Value offset) const {
  return impl_->ForwardDecoder(
      std::move(tokens), std::move(n_layer_self_k_cache),
      std::move(n_layer_self_v_cache), std::move(n_layer_cross_k),
      std::move(n_layer_cross_v), std::move(offset));
}

std::pair<Ort::Value, Ort::Value>
OfflineFireRedAsrModel::GetInitialSelfKVCache() const {
  return impl_->GetInitialSelfKVCache();
}

OrtAllocator *OfflineFireRedAsrModel::Allocator() const {
  return impl_->Allocator();
}

const OfflineFireRedAsrModelMetaData &OfflineFireRedAsrModel::GetModelMetadata()
    const {
  return impl_->GetModelMetadata();
}

#if __ANDROID_API__ >= 9
template OfflineFireRedAsrModel::OfflineFireRedAsrModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineFireRedAsrModel::OfflineFireRedAsrModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
