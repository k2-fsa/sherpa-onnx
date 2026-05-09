// sherpa-onnx/csrc/online-transducer-nemo-parakeet-unified-model.cc
//
// Copyright (c)  2026  Milan Leonard

#include "sherpa-onnx/csrc/online-transducer-nemo-parakeet-unified-model.h"

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

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

static constexpr const char *kStreamingModelType =
    "nemo_parakeet_unified_streaming";

class OnlineTransducerNeMoParakeetUnifiedModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.encoder), sess_opts_);
    InitEncoder(nullptr, 0);

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.decoder), sess_opts_);
    InitDecoder(nullptr, 0);

    joiner_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.joiner), sess_opts_);
    InitJoiner(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
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

  std::vector<Ort::Value> RunEncoder(Ort::Value features,
                                     Ort::Value features_length) {
    // (B, T, C) -> (B, C, T)
    features = Transpose12(allocator_, &features);

    std::array<Ort::Value, 2> inputs = {std::move(features),
                                        std::move(features_length)};

    return encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> RunDecoder(
      Ort::Value targets, std::vector<Ort::Value> states) {
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto shape = targets.GetTensorTypeAndShapeInfo().GetShape();
    int32_t batch_size = static_cast<int32_t>(shape[0]);

    std::vector<int64_t> length_shape = {batch_size};
    std::vector<int32_t> length_value(batch_size, 1);

    Ort::Value targets_length = Ort::Value::CreateTensor<int32_t>(
        memory_info, length_value.data(), batch_size, length_shape.data(),
        length_shape.size());

    std::vector<Ort::Value> decoder_inputs;
    decoder_inputs.reserve(2 + states.size());
    decoder_inputs.push_back(std::move(targets));
    decoder_inputs.push_back(std::move(targets_length));

    for (auto &s : states) {
      decoder_inputs.push_back(std::move(s));
    }

    auto decoder_out = decoder_sess_->Run(
        {}, decoder_input_names_ptr_.data(), decoder_inputs.data(),
        decoder_inputs.size(), decoder_output_names_ptr_.data(),
        decoder_output_names_ptr_.size());

    std::vector<Ort::Value> states_next;
    states_next.reserve(states.size());

    for (size_t i = 0; i != states.size(); ++i) {
      states_next.push_back(std::move(decoder_out[i + 2]));
    }

    return {std::move(decoder_out[0]), std::move(states_next)};
  }

  std::vector<Ort::Value> GetDecoderInitStates() {
    std::vector<Ort::Value> ans;
    ans.reserve(2);
    ans.push_back(View(&lstm0_));
    ans.push_back(View(&lstm1_));

    return ans;
  }

  Ort::Value RunJoiner(Ort::Value encoder_out, Ort::Value decoder_out) {
    std::array<Ort::Value, 2> joiner_input = {std::move(encoder_out),
                                              std::move(decoder_out)};
    auto logit = joiner_sess_->Run({}, joiner_input_names_ptr_.data(),
                                   joiner_input.data(), joiner_input.size(),
                                   joiner_output_names_ptr_.data(),
                                   joiner_output_names_ptr_.size());

    return std::move(logit[0]);
  }

  int32_t LeftFeatureFrames() const { return left_feature_frames_; }
  int32_t ChunkFeatureFrames() const { return chunk_feature_frames_; }
  int32_t RightFeatureFrames() const { return right_feature_frames_; }
  int32_t TotalFeatureFrames() const {
    return left_feature_frames_ + chunk_feature_frames_ + right_feature_frames_;
  }

  int32_t LeftEncoderFrames() const { return left_encoder_frames_; }
  int32_t ChunkEncoderFrames() const { return chunk_encoder_frames_; }
  int32_t RightEncoderFrames() const { return right_encoder_frames_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }
  int32_t VocabSize() const { return vocab_size_; }
  int32_t FeatureDim() const { return feat_dim_; }

  OrtAllocator *Allocator() { return allocator_; }

  std::string FeatureNormalizationMethod() const { return normalize_type_; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!encoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass model data or initialize the encoder session outside of "
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
      os << "---nemo parakeet unified encoder---\n";
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    std::string streaming_model_type;
    SHERPA_ONNX_READ_META_DATA_STR(streaming_model_type,
                                   "streaming_model_type");
    if (streaming_model_type != kStreamingModelType) {
      SHERPA_ONNX_LOGE("Expected streaming_model_type=%s, got: %s",
                       kStreamingModelType, streaming_model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    vocab_size_ += 1;

    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA_STR_ALLOW_EMPTY(normalize_type_,
                                               "normalize_type");
    SHERPA_ONNX_READ_META_DATA(pred_rnn_layers_, "pred_rnn_layers");
    SHERPA_ONNX_READ_META_DATA(pred_hidden_, "pred_hidden");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(feat_dim_, "feat_dim", -1);

    SHERPA_ONNX_READ_META_DATA(left_feature_frames_, "left_feature_frames");
    SHERPA_ONNX_READ_META_DATA(chunk_feature_frames_, "chunk_feature_frames");
    SHERPA_ONNX_READ_META_DATA(right_feature_frames_, "right_feature_frames");
    SHERPA_ONNX_READ_META_DATA(left_encoder_frames_, "left_encoder_frames");
    SHERPA_ONNX_READ_META_DATA(chunk_encoder_frames_, "chunk_encoder_frames");
    SHERPA_ONNX_READ_META_DATA(right_encoder_frames_, "right_encoder_frames");

    if (normalize_type_ == "NA") {
      normalize_type_ = "";
    }

    if (feat_dim_ <= 0) {
      feat_dim_ = encoder_sess_->GetInputTypeInfo(0)
                      .GetTensorTypeAndShapeInfo()
                      .GetShape()[1];
    }

    ValidateMetadata();
  }

  void ValidateMetadata() const {
    if (vocab_size_ <= 1) {
      SHERPA_ONNX_LOGE("Invalid vocab_size: %d", vocab_size_);
      SHERPA_ONNX_EXIT(-1);
    }
    if (subsampling_factor_ <= 0) {
      SHERPA_ONNX_LOGE("Invalid subsampling_factor: %d", subsampling_factor_);
      SHERPA_ONNX_EXIT(-1);
    }
    if (feat_dim_ <= 0) {
      SHERPA_ONNX_LOGE(
          "Could not determine feat_dim from metadata or encoder input shape "
          "(got %d). Set feat_dim in the encoder ONNX metadata.",
          feat_dim_);
      SHERPA_ONNX_EXIT(-1);
    }
    if (pred_rnn_layers_ <= 0 || pred_hidden_ <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid decoder state metadata: pred_rnn_layers=%d, "
          "pred_hidden=%d",
          pred_rnn_layers_, pred_hidden_);
      SHERPA_ONNX_EXIT(-1);
    }
    if (left_feature_frames_ < 0 || right_feature_frames_ < 0 ||
        left_encoder_frames_ < 0 || right_encoder_frames_ < 0) {
      SHERPA_ONNX_LOGE("Left/right context frame counts must be nonnegative");
      SHERPA_ONNX_EXIT(-1);
    }
    if (chunk_feature_frames_ <= 0 || chunk_encoder_frames_ <= 0) {
      SHERPA_ONNX_LOGE(
          "Chunk frame counts must be positive: feature=%d, "
          "encoder=%d",
          chunk_feature_frames_, chunk_encoder_frames_);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      decoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!decoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass model data or initialize the decoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);
    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);

    InitDecoderStates();
  }

  void InitDecoderStates() {
    std::array<int64_t, 3> s0_shape{pred_rnn_layers_, 1, pred_hidden_};
    lstm0_ = Ort::Value::CreateTensor<float>(allocator_, s0_shape.data(),
                                             s0_shape.size());
    Fill<float>(&lstm0_, 0);

    std::array<int64_t, 3> s1_shape{pred_rnn_layers_, 1, pred_hidden_};
    lstm1_ = Ort::Value::CreateTensor<float>(allocator_, s1_shape.data(),
                                             s1_shape.size());
    Fill<float>(&lstm1_, 0);
  }

  void InitJoiner(void *model_data, size_t model_data_length) {
    if (model_data) {
      joiner_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!joiner_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass model data or initialize the joiner session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(joiner_sess_.get(), &joiner_input_names_,
                  &joiner_input_names_ptr_);
    GetOutputNames(joiner_sess_.get(), &joiner_output_names_,
                   &joiner_output_names_ptr_);
  }

 private:
  OnlineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;
  std::unique_ptr<Ort::Session> joiner_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;
  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;
  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<std::string> joiner_input_names_;
  std::vector<const char *> joiner_input_names_ptr_;
  std::vector<std::string> joiner_output_names_;
  std::vector<const char *> joiner_output_names_ptr_;

  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 8;
  int32_t feat_dim_ = 128;
  std::string normalize_type_;
  int32_t pred_rnn_layers_ = -1;
  int32_t pred_hidden_ = -1;

  int32_t left_feature_frames_ = 0;
  int32_t chunk_feature_frames_ = 0;
  int32_t right_feature_frames_ = 0;
  int32_t left_encoder_frames_ = 0;
  int32_t chunk_encoder_frames_ = 0;
  int32_t right_encoder_frames_ = 0;

  Ort::Value lstm0_{nullptr};
  Ort::Value lstm1_{nullptr};
};

OnlineTransducerNeMoParakeetUnifiedModel::
    OnlineTransducerNeMoParakeetUnifiedModel(const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineTransducerNeMoParakeetUnifiedModel::
    OnlineTransducerNeMoParakeetUnifiedModel(Manager *mgr,
                                             const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OnlineTransducerNeMoParakeetUnifiedModel::
    ~OnlineTransducerNeMoParakeetUnifiedModel() = default;

std::vector<Ort::Value> OnlineTransducerNeMoParakeetUnifiedModel::RunEncoder(
    Ort::Value features, Ort::Value features_length) const {
  return impl_->RunEncoder(std::move(features), std::move(features_length));
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineTransducerNeMoParakeetUnifiedModel::RunDecoder(
    Ort::Value targets, std::vector<Ort::Value> states) const {
  return impl_->RunDecoder(std::move(targets), std::move(states));
}

std::vector<Ort::Value>
OnlineTransducerNeMoParakeetUnifiedModel::GetDecoderInitStates() const {
  return impl_->GetDecoderInitStates();
}

Ort::Value OnlineTransducerNeMoParakeetUnifiedModel::RunJoiner(
    Ort::Value encoder_out, Ort::Value decoder_out) const {
  return impl_->RunJoiner(std::move(encoder_out), std::move(decoder_out));
}

int32_t OnlineTransducerNeMoParakeetUnifiedModel::LeftFeatureFrames() const {
  return impl_->LeftFeatureFrames();
}

int32_t OnlineTransducerNeMoParakeetUnifiedModel::ChunkFeatureFrames() const {
  return impl_->ChunkFeatureFrames();
}

int32_t OnlineTransducerNeMoParakeetUnifiedModel::RightFeatureFrames() const {
  return impl_->RightFeatureFrames();
}

int32_t OnlineTransducerNeMoParakeetUnifiedModel::TotalFeatureFrames() const {
  return impl_->TotalFeatureFrames();
}

int32_t OnlineTransducerNeMoParakeetUnifiedModel::LeftEncoderFrames() const {
  return impl_->LeftEncoderFrames();
}

int32_t OnlineTransducerNeMoParakeetUnifiedModel::ChunkEncoderFrames() const {
  return impl_->ChunkEncoderFrames();
}

int32_t OnlineTransducerNeMoParakeetUnifiedModel::RightEncoderFrames() const {
  return impl_->RightEncoderFrames();
}

int32_t OnlineTransducerNeMoParakeetUnifiedModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

int32_t OnlineTransducerNeMoParakeetUnifiedModel::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OnlineTransducerNeMoParakeetUnifiedModel::FeatureDim() const {
  return impl_->FeatureDim();
}

OrtAllocator *OnlineTransducerNeMoParakeetUnifiedModel::Allocator() const {
  return impl_->Allocator();
}

std::string
OnlineTransducerNeMoParakeetUnifiedModel::FeatureNormalizationMethod() const {
  return impl_->FeatureNormalizationMethod();
}

#if __ANDROID_API__ >= 9
template OnlineTransducerNeMoParakeetUnifiedModel::
    OnlineTransducerNeMoParakeetUnifiedModel(AAssetManager *mgr,
                                             const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineTransducerNeMoParakeetUnifiedModel::
    OnlineTransducerNeMoParakeetUnifiedModel(NativeResourceManager *mgr,
                                             const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
