// sherpa-onnx/csrc/offline-dolphin-attention-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-dolphin-attention-model.h"

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

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineDolphinAttentionModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.dolphin.encoder), sess_opts_);
    InitEncoder(nullptr, 0);

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.dolphin.decoder), sess_opts_);
    InitDecoder(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    {
      auto buf = ReadFile(mgr, config_.dolphin.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config_.dolphin.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  Ort::Value ForwardEncoder(Ort::Value features, Ort::Value features_length) {
    std::array<Ort::Value, 2> inputs = {
        std::move(features),
        std::move(features_length),
    };

    auto outs = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());
    return std::move(outs[0]);
  }

  Ort::Value ForwardDecoderStep(Ort::Value &encoder_out, Ort::Value ys) {
    // Wrap encoder_out's buffer in a non-owning tensor so it can be reused
    // across all decoder steps.
    auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
    float *p = encoder_out.GetTensorMutableData<float>();
    int64_t num_elems = 1;
    for (auto s : shape) {
      num_elems *= s;
    }

    Ort::Value encoder_out_view = Ort::Value::CreateTensor<float>(
        cpu_mem_info_, p, num_elems, shape.data(), shape.size());

    std::array<Ort::Value, 2> inputs = {
        std::move(encoder_out_view),
        std::move(ys),
    };

    auto outs = decoder_sess_->Run(
        {}, decoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());
    return std::move(outs[0]);
  }

  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const {
    using RowMajorMat =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<RowMajorMat> x(features, num_frames, feat_dim);

    Eigen::Map<const Eigen::RowVectorXf> mean(meta_data_.mean.data(), feat_dim);
    Eigen::Map<const Eigen::RowVectorXf> inv_std(meta_data_.inv_stddev.data(),
                                                 feat_dim);
    x.array() =
        (x.array().rowwise() - mean.array()).rowwise() * inv_std.array();
  }

  int32_t VocabSize() const { return meta_data_.vocab_size; }

  OrtAllocator *Allocator() { return allocator_; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    }

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);
    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.mean, "mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_data_.inv_stddev, "invstd");
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      decoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    }

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);
    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);

    Ort::ModelMetadata meta_data = decoder_sess_->GetModelMetadata();

    Ort::AllocatorWithDefaultOptions allocator;
    SHERPA_ONNX_READ_META_DATA(meta_data_.vocab_size, "vocab_size");
    SHERPA_ONNX_READ_META_DATA(meta_data_.sos, "sos");
    SHERPA_ONNX_READ_META_DATA(meta_data_.eos, "eos");
  }

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo cpu_mem_info_;

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

  OfflineDolphinModelMetaData meta_data_;
};

OfflineDolphinAttentionModel::OfflineDolphinAttentionModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineDolphinAttentionModel::OfflineDolphinAttentionModel(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineDolphinAttentionModel::~OfflineDolphinAttentionModel() = default;

Ort::Value OfflineDolphinAttentionModel::ForwardEncoder(
    Ort::Value features, Ort::Value features_length) const {
  return impl_->ForwardEncoder(std::move(features), std::move(features_length));
}

Ort::Value OfflineDolphinAttentionModel::ForwardDecoderStep(
    Ort::Value &encoder_out, Ort::Value ys) const {
  return impl_->ForwardDecoderStep(encoder_out, std::move(ys));
}

void OfflineDolphinAttentionModel::NormalizeFeatures(float *features,
                                                     int32_t num_frames,
                                                     int32_t feat_dim) const {
  impl_->NormalizeFeatures(features, num_frames, feat_dim);
}

int32_t OfflineDolphinAttentionModel::VocabSize() const {
  return impl_->VocabSize();
}

OrtAllocator *OfflineDolphinAttentionModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineDolphinAttentionModel::OfflineDolphinAttentionModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineDolphinAttentionModel::OfflineDolphinAttentionModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
