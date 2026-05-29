// sherpa-onnx/csrc/offline-catt-model.cc
//
// Copyright (c)  2026  Matias Lin
#include "sherpa-onnx/csrc/offline-catt-model.h"

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

class OfflineCATTModel::Impl {
 public:
  explicit Impl(const OfflineDiacritizationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.catt_encoder), sess_opts_);
    InitEncoder();

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.catt_decoder), sess_opts_);
    InitDecoder();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineDiacritizationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config_.catt_encoder);
      encoder_sess_ = std::make_unique<Ort::Session>(env_, buf.data(),
                                                     buf.size(), sess_opts_);
      InitEncoder();
    }
    {
      auto buf = ReadFile(mgr, config_.catt_decoder);
      decoder_sess_ = std::make_unique<Ort::Session>(env_, buf.data(),
                                                     buf.size(), sess_opts_);
      InitDecoder();
    }
  }

  Ort::Value RunEncoder(Ort::Value src, Ort::Value src_mask) const {
    std::array<Ort::Value, 2> inputs = {std::move(src), std::move(src_mask)};
    auto ans = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());
    return std::move(ans[0]);
  }

  Ort::Value RunDecoder(Ort::Value enc_src) const {
    auto ans = decoder_sess_->Run({}, decoder_input_names_ptr_.data(), &enc_src,
                                  1, decoder_output_names_ptr_.data(),
                                  decoder_output_names_ptr_.size());
    return std::move(ans[0]);
  }

  OrtAllocator *Allocator() const { return allocator_; }

 private:
  void InitEncoder() {
    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);
    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    if (config_.debug) {
      std::ostringstream os;
      os << "CATT encoder inputs: ";
      for (const auto &n : encoder_input_names_) os << n << " ";
      os << "\nCATT encoder outputs: ";
      for (const auto &n : encoder_output_names_) os << n << " ";
      SHERPA_ONNX_LOGE("\n%s\n", os.str().c_str());
    }
  }

  void InitDecoder() {
    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);
    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);

    if (config_.debug) {
      std::ostringstream os;
      os << "CATT decoder inputs: ";
      for (const auto &n : decoder_input_names_) os << n << " ";
      os << "\nCATT decoder outputs: ";
      for (const auto &n : decoder_output_names_) os << n << " ";
      SHERPA_ONNX_LOGE("\n%s\n", os.str().c_str());
    }
  }

  OfflineDiacritizationModelConfig config_;
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

OfflineCATTModel::OfflineCATTModel(
    const OfflineDiacritizationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineCATTModel::OfflineCATTModel(
    Manager *mgr, const OfflineDiacritizationModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

#if __ANDROID_API__ >= 9
template OfflineCATTModel::OfflineCATTModel(
    AAssetManager *mgr, const OfflineDiacritizationModelConfig &config);
#endif

#if __OHOS__
template OfflineCATTModel::OfflineCATTModel(
    NativeResourceManager *mgr, const OfflineDiacritizationModelConfig &config);
#endif

OfflineCATTModel::~OfflineCATTModel() = default;

Ort::Value OfflineCATTModel::RunEncoder(Ort::Value src, Ort::Value src_mask) const {
  return impl_->RunEncoder(std::move(src), std::move(src_mask));
}

Ort::Value OfflineCATTModel::RunDecoder(Ort::Value enc_src) const {
  return impl_->RunDecoder(std::move(enc_src));
}

OrtAllocator *OfflineCATTModel::Allocator() const { return impl_->Allocator(); }

}  // namespace sherpa_onnx
