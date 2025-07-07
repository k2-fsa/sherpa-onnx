// sherpa-onnx/csrc/offline-canary-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-canary-model.h"

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

class OfflineCanaryModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.canary.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.canary.decoder);
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
      auto buf = ReadFile(mgr, config.canary.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.canary.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  std::vector<Ort::Value> ForwardEncoder(Ort::Value features,
                                         Ort::Value features_length) {}

  std::pair<Ort::Value, std::vector<Ort::Value>> ForwardDecoder(
      Ort::Value tokens, std::vector<Ort::Value> decoder_states,
      Ort::Value encoder_states, Ort::Value enc_mask) {}

  std::vector<Ort::Value> GetInitialDecoderStates() {
    std::array<int64_t, 3> shape{1, 0, 1024};

    std::vector<Ort::Value> ans;
    ans.reserve(6);
    for (int32_t i = 0; i < 6; ++i) {
      Ort::Value state = Ort::Value::CreateTensor<float>(
          Allocator(), shape.data(), shape.size());

      ans.push_back(std::move(state));
    }

    return ans;
  }

  OrtAllocator *Allocator() { return allocator_; }

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
};

OfflineCanaryModel::OfflineCanaryModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineCanaryModel::OfflineCanaryModel(Manager *mgr,
                                       const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineCanaryModel::~OfflineCanaryModel() = default;

std::vector<Ort::Value> OfflineCanaryModel::ForwardEncoder(
    Ort::Value features, Ort::Value features_length) const {
  return impl_->ForwardEncoder(std::move(features), std::move(features_length));
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OfflineCanaryModel::ForwardDecoder(Ort::Value tokens,
                                   std::vector<Ort::Value> decoder_states,
                                   Ort::Value encoder_states,
                                   Ort::Value enc_mask) const {
  return impl_->ForwardDecoder(std::move(tokens), std::move(decoder_states),
                               std::move(encoder_states), std::move(enc_mask));
}

std::vector<Ort::Value> OfflineCanaryModel::GetInitialDecoderStates() const {
  return impl_->GetInitialDecoderStates();
}

OrtAllocator *OfflineCanaryModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineCanaryModel::OfflineCanaryModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineCanaryModel::OfflineCanaryModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
