// sherpa-onnx/csrc/offline-qwen3-forced-aligner-model.cc
//
// Copyright (c)  2026  losewayy

#include "sherpa-onnx/csrc/offline-qwen3-forced-aligner-model.h"

#include <array>
#include <memory>
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

class OfflineQwen3ForcedAlignerModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : env_(ORT_LOGGING_LEVEL_ERROR, "qwen3-forced-aligner"),
        sess_opts_conv_(GetSessionOptions(config)),
        sess_opts_encoder_(GetSessionOptions(config)),
        sess_opts_decoder_(GetSessionOptions(config)),
        allocator_() {
    const auto &c = config.qwen3_asr;

    conv_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(c.forced_aligner_conv_frontend),
        sess_opts_conv_);
    GetInputNames(conv_sess_.get(), &conv_input_names_, &conv_input_names_ptr_);
    GetOutputNames(conv_sess_.get(), &conv_output_names_,
                   &conv_output_names_ptr_);

    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(c.forced_aligner_encoder),
        sess_opts_encoder_);
    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);
    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(c.forced_aligner_decoder),
        sess_opts_decoder_);
    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);
    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : env_(ORT_LOGGING_LEVEL_ERROR, "qwen3-forced-aligner"),
        sess_opts_conv_(GetSessionOptions(config)),
        sess_opts_encoder_(GetSessionOptions(config)),
        sess_opts_decoder_(GetSessionOptions(config)),
        allocator_() {
    const auto &c = config.qwen3_asr;

    {
      auto buf = ReadFile(mgr, c.forced_aligner_conv_frontend);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read forced_aligner_conv_frontend: %s",
                         c.forced_aligner_conv_frontend.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      conv_sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                                  sess_opts_conv_);
      GetInputNames(conv_sess_.get(), &conv_input_names_,
                    &conv_input_names_ptr_);
      GetOutputNames(conv_sess_.get(), &conv_output_names_,
                     &conv_output_names_ptr_);
    }
    {
      auto buf = ReadFile(mgr, c.forced_aligner_encoder);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read forced_aligner_encoder: %s",
                         c.forced_aligner_encoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, buf.data(), buf.size(), sess_opts_encoder_);
      GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                    &encoder_input_names_ptr_);
      GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                     &encoder_output_names_ptr_);
    }
    {
      auto buf = ReadFile(mgr, c.forced_aligner_decoder);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read forced_aligner_decoder: %s",
                         c.forced_aligner_decoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      decoder_sess_ = std::make_unique<Ort::Session>(
          env_, buf.data(), buf.size(), sess_opts_decoder_);
      GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                    &decoder_input_names_ptr_);
      GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                     &decoder_output_names_ptr_);
    }
  }

  Ort::Value ForwardConvFrontend(Ort::Value input_features) {
    std::array<Ort::Value, 1> inputs = {std::move(input_features)};
    auto outputs = conv_sess_->Run(Ort::RunOptions{nullptr},
                                   conv_input_names_ptr_.data(), inputs.data(),
                                   inputs.size(), conv_output_names_ptr_.data(),
                                   conv_output_names_ptr_.size());
    if (outputs.empty() || !outputs[0].IsTensor()) {
      SHERPA_ONNX_LOGE(
          "qwen3-forced-aligner: ForwardConvFrontend got no tensor output");
      SHERPA_ONNX_EXIT(-1);
    }
    return std::move(outputs[0]);
  }

  Ort::Value ForwardEncoder(Ort::Value conv_output,
                            Ort::Value feature_attention_mask) {
    std::array<Ort::Value, 2> inputs = {std::move(conv_output),
                                        std::move(feature_attention_mask)};
    auto outputs = encoder_sess_->Run(
        Ort::RunOptions{nullptr}, encoder_input_names_ptr_.data(),
        inputs.data(), inputs.size(), encoder_output_names_ptr_.data(),
        encoder_output_names_ptr_.size());
    if (outputs.empty() || !outputs[0].IsTensor()) {
      SHERPA_ONNX_LOGE(
          "qwen3-forced-aligner: ForwardEncoder got no tensor output");
      SHERPA_ONNX_EXIT(-1);
    }
    return std::move(outputs[0]);
  }

  Ort::Value ForwardDecoder(Ort::Value input_ids, Ort::Value audio_features,
                            Ort::Value attention_mask) {
    std::array<Ort::Value, 3> inputs = {std::move(input_ids),
                                        std::move(audio_features),
                                        std::move(attention_mask)};
    auto outputs = decoder_sess_->Run(
        Ort::RunOptions{nullptr}, decoder_input_names_ptr_.data(),
        inputs.data(), inputs.size(), decoder_output_names_ptr_.data(),
        decoder_output_names_ptr_.size());
    if (outputs.empty() || !outputs[0].IsTensor()) {
      SHERPA_ONNX_LOGE(
          "qwen3-forced-aligner: ForwardDecoder got no tensor output");
      SHERPA_ONNX_EXIT(-1);
    }
    return std::move(outputs[0]);
  }

  OrtAllocator *Allocator() { return allocator_; }

 private:
  Ort::Env env_;
  Ort::SessionOptions sess_opts_conv_;
  Ort::SessionOptions sess_opts_encoder_;
  Ort::SessionOptions sess_opts_decoder_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> conv_sess_;
  std::vector<std::string> conv_input_names_;
  std::vector<const char *> conv_input_names_ptr_;
  std::vector<std::string> conv_output_names_;
  std::vector<const char *> conv_output_names_ptr_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;
  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::unique_ptr<Ort::Session> decoder_sess_;
  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;
  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;
};

OfflineQwen3ForcedAlignerModel::OfflineQwen3ForcedAlignerModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineQwen3ForcedAlignerModel::OfflineQwen3ForcedAlignerModel(
    AAssetManager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

#if __OHOS__
OfflineQwen3ForcedAlignerModel::OfflineQwen3ForcedAlignerModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineQwen3ForcedAlignerModel::~OfflineQwen3ForcedAlignerModel() = default;

Ort::Value OfflineQwen3ForcedAlignerModel::ForwardConvFrontend(
    Ort::Value input_features) const {
  return impl_->ForwardConvFrontend(std::move(input_features));
}

Ort::Value OfflineQwen3ForcedAlignerModel::ForwardEncoder(
    Ort::Value conv_output, Ort::Value feature_attention_mask) const {
  return impl_->ForwardEncoder(std::move(conv_output),
                               std::move(feature_attention_mask));
}

Ort::Value OfflineQwen3ForcedAlignerModel::ForwardDecoder(
    Ort::Value input_ids, Ort::Value audio_features,
    Ort::Value attention_mask) const {
  return impl_->ForwardDecoder(std::move(input_ids), std::move(audio_features),
                               std::move(attention_mask));
}

OrtAllocator *OfflineQwen3ForcedAlignerModel::Allocator() const {
  return impl_->Allocator();
}

}  // namespace sherpa_onnx
