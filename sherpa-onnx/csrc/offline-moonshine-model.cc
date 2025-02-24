// sherpa-onnx/csrc/offline-moonshine-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-moonshine-model.h"

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

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineMoonshineModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.moonshine.preprocessor);
      InitPreprocessor(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.moonshine.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.moonshine.uncached_decoder);
      InitUnCachedDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.moonshine.cached_decoder);
      InitCachedDecoder(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.moonshine.preprocessor);
      InitPreprocessor(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.moonshine.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.moonshine.uncached_decoder);
      InitUnCachedDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.moonshine.cached_decoder);
      InitCachedDecoder(buf.data(), buf.size());
    }
  }

  Ort::Value ForwardPreprocessor(Ort::Value audio) {
    auto features = preprocessor_sess_->Run(
        {}, preprocessor_input_names_ptr_.data(), &audio, 1,
        preprocessor_output_names_ptr_.data(),
        preprocessor_output_names_ptr_.size());

    return std::move(features[0]);
  }

  Ort::Value ForwardEncoder(Ort::Value features, Ort::Value features_len) {
    std::array<Ort::Value, 2> encoder_inputs{std::move(features),
                                             std::move(features_len)};
    auto encoder_out = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), encoder_inputs.data(),
        encoder_inputs.size(), encoder_output_names_ptr_.data(),
        encoder_output_names_ptr_.size());

    return std::move(encoder_out[0]);
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> ForwardUnCachedDecoder(
      Ort::Value tokens, Ort::Value seq_len, Ort::Value encoder_out) {
    std::array<Ort::Value, 3> uncached_decoder_input = {
        std::move(tokens),
        std::move(encoder_out),
        std::move(seq_len),
    };

    auto uncached_decoder_out = uncached_decoder_sess_->Run(
        {}, uncached_decoder_input_names_ptr_.data(),
        uncached_decoder_input.data(), uncached_decoder_input.size(),
        uncached_decoder_output_names_ptr_.data(),
        uncached_decoder_output_names_ptr_.size());

    std::vector<Ort::Value> states;
    states.reserve(uncached_decoder_out.size() - 1);

    int32_t i = -1;
    for (auto &s : uncached_decoder_out) {
      ++i;
      if (i == 0) {
        continue;
      }

      states.push_back(std::move(s));
    }

    return {std::move(uncached_decoder_out[0]), std::move(states)};
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> ForwardCachedDecoder(
      Ort::Value tokens, Ort::Value seq_len, Ort::Value encoder_out,
      std::vector<Ort::Value> states) {
    std::vector<Ort::Value> cached_decoder_input;
    cached_decoder_input.reserve(3 + states.size());
    cached_decoder_input.push_back(std::move(tokens));
    cached_decoder_input.push_back(std::move(encoder_out));
    cached_decoder_input.push_back(std::move(seq_len));

    for (auto &s : states) {
      cached_decoder_input.push_back(std::move(s));
    }

    auto cached_decoder_out = cached_decoder_sess_->Run(
        {}, cached_decoder_input_names_ptr_.data(), cached_decoder_input.data(),
        cached_decoder_input.size(), cached_decoder_output_names_ptr_.data(),
        cached_decoder_output_names_ptr_.size());

    std::vector<Ort::Value> next_states;
    next_states.reserve(cached_decoder_out.size() - 1);

    int32_t i = -1;
    for (auto &s : cached_decoder_out) {
      ++i;
      if (i == 0) {
        continue;
      }

      next_states.push_back(std::move(s));
    }

    return {std::move(cached_decoder_out[0]), std::move(next_states)};
  }

  OrtAllocator *Allocator() { return allocator_; }

 private:
  void InitPreprocessor(void *model_data, size_t model_data_length) {
    preprocessor_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(preprocessor_sess_.get(), &preprocessor_input_names_,
                  &preprocessor_input_names_ptr_);

    GetOutputNames(preprocessor_sess_.get(), &preprocessor_output_names_,
                   &preprocessor_output_names_ptr_);
  }

  void InitEncoder(void *model_data, size_t model_data_length) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);
  }

  void InitUnCachedDecoder(void *model_data, size_t model_data_length) {
    uncached_decoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(uncached_decoder_sess_.get(), &uncached_decoder_input_names_,
                  &uncached_decoder_input_names_ptr_);

    GetOutputNames(uncached_decoder_sess_.get(),
                   &uncached_decoder_output_names_,
                   &uncached_decoder_output_names_ptr_);
  }

  void InitCachedDecoder(void *model_data, size_t model_data_length) {
    cached_decoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(cached_decoder_sess_.get(), &cached_decoder_input_names_,
                  &cached_decoder_input_names_ptr_);

    GetOutputNames(cached_decoder_sess_.get(), &cached_decoder_output_names_,
                   &cached_decoder_output_names_ptr_);
  }

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> preprocessor_sess_;
  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> uncached_decoder_sess_;
  std::unique_ptr<Ort::Session> cached_decoder_sess_;

  std::vector<std::string> preprocessor_input_names_;
  std::vector<const char *> preprocessor_input_names_ptr_;

  std::vector<std::string> preprocessor_output_names_;
  std::vector<const char *> preprocessor_output_names_ptr_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> uncached_decoder_input_names_;
  std::vector<const char *> uncached_decoder_input_names_ptr_;

  std::vector<std::string> uncached_decoder_output_names_;
  std::vector<const char *> uncached_decoder_output_names_ptr_;

  std::vector<std::string> cached_decoder_input_names_;
  std::vector<const char *> cached_decoder_input_names_ptr_;

  std::vector<std::string> cached_decoder_output_names_;
  std::vector<const char *> cached_decoder_output_names_ptr_;
};

OfflineMoonshineModel::OfflineMoonshineModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineMoonshineModel::OfflineMoonshineModel(Manager *mgr,
                                             const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineMoonshineModel::~OfflineMoonshineModel() = default;

Ort::Value OfflineMoonshineModel::ForwardPreprocessor(Ort::Value audio) const {
  return impl_->ForwardPreprocessor(std::move(audio));
}

Ort::Value OfflineMoonshineModel::ForwardEncoder(
    Ort::Value features, Ort::Value features_len) const {
  return impl_->ForwardEncoder(std::move(features), std::move(features_len));
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OfflineMoonshineModel::ForwardUnCachedDecoder(Ort::Value token,
                                              Ort::Value seq_len,
                                              Ort::Value encoder_out) const {
  return impl_->ForwardUnCachedDecoder(std::move(token), std::move(seq_len),
                                       std::move(encoder_out));
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OfflineMoonshineModel::ForwardCachedDecoder(
    Ort::Value token, Ort::Value seq_len, Ort::Value encoder_out,
    std::vector<Ort::Value> states) const {
  return impl_->ForwardCachedDecoder(std::move(token), std::move(seq_len),
                                     std::move(encoder_out), std::move(states));
}

OrtAllocator *OfflineMoonshineModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineMoonshineModel::OfflineMoonshineModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineMoonshineModel::OfflineMoonshineModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
