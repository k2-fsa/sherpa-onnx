// sherpa-onnx/csrc/rknn/offline-paraformer-model-rknn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/offline-paraformer-model-rknn.h"

#include <algorithm>
#include <array>
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
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/rknn/context-blocking-queue-rknn.h"
#include "sherpa-onnx/csrc/rknn/macros.h"
#include "sherpa-onnx/csrc/rknn/utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineParaformerModelRknn::Impl {
 public:
  ~Impl() {
    auto ret = rknn_destroy(encoder_ctx_);
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE("Failed to destroy the encoder context");
    }

    ret = rknn_destroy(predictor_ctx_);
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE("Failed to destroy the predictor context");
    }

    ret = rknn_destroy(decoder_ctx_);
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE("Failed to destroy the decoder context");
    }
  }

  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    std::vector<std::string> filenames;
    SplitStringToVector(config_.paraformer.model, ",", false, &filenames);
    if (filenames.size() != 3) {
      SHERPA_ONNX_LOGE("Invalid paraformer ascend NPU model '%s'",
                       config_.paraformer.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    {
      auto buf = ReadFile(filenames[0]);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(filenames[1]);
      InitPredictor(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(filenames[2]);
      InitDecoder(buf.data(), buf.size());
    }

    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    std::vector<std::string> filenames;
    SplitStringToVector(config_.paraformer.model, ",", false, &filenames);
    if (filenames.size() != 3) {
      SHERPA_ONNX_LOGE("Invalid paraformer ascend NPU model '%s'",
                       config_.paraformer.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    {
      auto buf = ReadFile(mgr, filenames[0]);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, filenames[1]);
      InitPredictor(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, filenames[2]);
      InitDecoder(buf.data(), buf.size());
    }

    PostInit();
  }

  std::vector<float> Run(std::vector<float> features) {
    std::vector<float> encoder_out = RunEncoder(features);
    if (encoder_out.empty()) {
      return {};
    }

    std::vector<float> alphas = RunPredictor(encoder_out);

    std::vector<float> acoustic_embedding =
        ComputeAcousticEmbedding(encoder_out, alphas, encoder_out_dim_);
    if (acoustic_embedding.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE("No speech found in the input audio");
      }

      return {};
    }

    int32_t num_tokens = acoustic_embedding.size() / encoder_out_dim_;

    acoustic_embedding.resize(encoder_out.size());

    return RunDecoder(std::move(encoder_out), std::move(acoustic_embedding),
                      num_tokens);
  }

  int32_t VocabSize() const { return vocab_size_; }

 private:
  std::vector<float> RunEncoder(std::vector<float> features) {
    features = ApplyLFR(std::move(features));
    if (features.empty()) {
      return {};
    }

    std::vector<rknn_input> inputs(encoder_input_attrs_.size());

    inputs[0].index = encoder_input_attrs_[0].index;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = encoder_input_attrs_[0].fmt;
    inputs[0].buf = reinterpret_cast<void *>(features.data());
    inputs[0].size = features.size() * sizeof(float);

    std::vector<float> out(encoder_output_attrs_[0].n_elems);

    std::vector<rknn_output> outputs(encoder_output_attrs_.size());
    outputs[0].index = encoder_output_attrs_[0].index;
    outputs[0].is_prealloc = 1;
    outputs[0].want_float = 1;
    outputs[0].size = out.size() * sizeof(float);
    outputs[0].buf = reinterpret_cast<void *>(out.data());

    rknn_context ctx = encoder_ctx_queue_->Take();

    auto ret = rknn_inputs_set(ctx, inputs.size(), inputs.data());
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set encoder inputs");

    ret = rknn_run(ctx, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run the encoder model");

    ret = rknn_outputs_get(ctx, outputs.size(), outputs.data(), nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get encoder output");

    encoder_ctx_queue_->Put(ctx);

    return out;
  }

  std::vector<float> RunPredictor(const std::vector<float> &encoder_out) {
    std::vector<rknn_input> inputs(predictor_input_attrs_.size());

    inputs[0].index = predictor_input_attrs_[0].index;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = predictor_input_attrs_[0].fmt;
    inputs[0].buf =
        reinterpret_cast<void *>(const_cast<float *>(encoder_out.data()));
    inputs[0].size = encoder_out.size() * sizeof(float);

    std::vector<float> out(predictor_output_attrs_[0].n_elems);

    std::vector<rknn_output> outputs(predictor_output_attrs_.size());
    outputs[0].index = predictor_output_attrs_[0].index;
    outputs[0].is_prealloc = 1;
    outputs[0].want_float = 1;
    outputs[0].size = out.size() * sizeof(float);
    outputs[0].buf = reinterpret_cast<void *>(out.data());

    rknn_context ctx = predictor_ctx_queue_->Take();

    auto ret = rknn_inputs_set(ctx, inputs.size(), inputs.data());
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set predictor inputs");

    ret = rknn_run(ctx, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run the predictor model");

    ret = rknn_outputs_get(ctx, outputs.size(), outputs.data(), nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get predictor output");

    predictor_ctx_queue_->Put(ctx);

    return out;
  }

  std::vector<float> RunDecoder(std::vector<float> encoder_out,
                                std::vector<float> acoustic_embedding,
                                int32_t num_tokens) {
    int32_t num_frames = encoder_out.size() / encoder_out_dim_;

    std::vector<rknn_input> inputs(decoder_input_attrs_.size());

    inputs[0].index = decoder_input_attrs_[0].index;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = decoder_input_attrs_[0].fmt;
    inputs[0].buf = reinterpret_cast<void *>(encoder_out.data());
    inputs[0].size = encoder_out.size() * sizeof(float);

    inputs[1].index = decoder_input_attrs_[1].index;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].fmt = decoder_input_attrs_[1].fmt;
    inputs[1].buf = reinterpret_cast<void *>(acoustic_embedding.data());
    inputs[1].size = acoustic_embedding.size() * sizeof(float);

    std::vector<float> mask(num_frames, 1);
    std::fill(mask.begin() + num_tokens, mask.end(), 0);

    inputs[2].index = decoder_input_attrs_[2].index;
    inputs[2].type = RKNN_TENSOR_FLOAT32;
    inputs[2].fmt = decoder_input_attrs_[2].fmt;
    inputs[2].buf = reinterpret_cast<void *>(mask.data());
    inputs[2].size = mask.size() * sizeof(float);

    std::vector<float> out(decoder_output_attrs_[0].n_elems);

    std::vector<rknn_output> outputs(decoder_output_attrs_.size());
    outputs[0].index = decoder_output_attrs_[0].index;
    outputs[0].is_prealloc = 1;
    outputs[0].want_float = 1;
    outputs[0].size = out.size() * sizeof(float);
    outputs[0].buf = reinterpret_cast<void *>(out.data());

    rknn_context ctx = decoder_ctx_queue_->Take();

    auto ret = rknn_inputs_set(ctx, inputs.size(), inputs.data());
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set decoder inputs");

    ret = rknn_run(ctx, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run the decoder model");

    ret = rknn_outputs_get(ctx, outputs.size(), outputs.data(), nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get decoder output");

    decoder_ctx_queue_->Put(ctx);

    return out;
  }

  void InitEncoder(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &encoder_ctx_);

    InitInputOutputAttrs(encoder_ctx_, config_.debug, &encoder_input_attrs_,
                         &encoder_output_attrs_);

    num_input_frames_ = encoder_input_attrs_[0].dims[1];
    encoder_out_dim_ = encoder_output_attrs_[0].dims[2];
    if (config_.debug) {
      SHERPA_ONNX_LOGE("num_input_frames_: %d", num_input_frames_);
      SHERPA_ONNX_LOGE("encoder_out_dim:: %d", encoder_out_dim_);
    }
  }

  void InitPredictor(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &predictor_ctx_);

    InitInputOutputAttrs(predictor_ctx_, config_.debug, &predictor_input_attrs_,
                         &predictor_output_attrs_);
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &decoder_ctx_);

    InitInputOutputAttrs(decoder_ctx_, config_.debug, &decoder_input_attrs_,
                         &decoder_output_attrs_);
    vocab_size_ = decoder_output_attrs_[0].dims[2];
    if (config_.debug) {
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
    }
  }

  std::vector<float> ApplyLFR(std::vector<float> in) const {
    int32_t lfr_window_size = 7;
    int32_t lfr_window_shift = 6;
    int32_t in_feat_dim = 80;

    int32_t in_num_frames = in.size() / in_feat_dim;
    if (in_num_frames < lfr_window_size) {
      return {};
    }

    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;

    if (out_num_frames > num_input_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          out_num_frames, num_input_frames_);

      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios.");

      out_num_frames = num_input_frames_;
    }

    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(num_input_frames_ * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  void PostInit() {
    if (config_.num_threads > 1) {
      config_.num_threads = 1;
    }

    encoder_ctx_queue_ = std::make_unique<ContextBlockingQueueRknn>(
        encoder_ctx_, config_.num_threads);

    predictor_ctx_queue_ = std::make_unique<ContextBlockingQueueRknn>(
        predictor_ctx_, config_.num_threads);

    decoder_ctx_queue_ = std::make_unique<ContextBlockingQueueRknn>(
        decoder_ctx_, config_.num_threads);
  }

 private:
  OfflineModelConfig config_;

  rknn_context encoder_ctx_ = 0;
  rknn_context predictor_ctx_ = 0;
  rknn_context decoder_ctx_ = 0;

  std::unique_ptr<ContextBlockingQueueRknn> encoder_ctx_queue_;
  std::unique_ptr<ContextBlockingQueueRknn> predictor_ctx_queue_;
  std::unique_ptr<ContextBlockingQueueRknn> decoder_ctx_queue_;

  std::vector<rknn_tensor_attr> encoder_input_attrs_;
  std::vector<rknn_tensor_attr> encoder_output_attrs_;

  std::vector<rknn_tensor_attr> predictor_input_attrs_;
  std::vector<rknn_tensor_attr> predictor_output_attrs_;

  std::vector<rknn_tensor_attr> decoder_input_attrs_;
  std::vector<rknn_tensor_attr> decoder_output_attrs_;

  int32_t vocab_size_ = 0;
  int32_t num_input_frames_ = -1;
  int32_t encoder_out_dim_ = -1;
};

OfflineParaformerModelRknn::~OfflineParaformerModelRknn() = default;

OfflineParaformerModelRknn::OfflineParaformerModelRknn(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineParaformerModelRknn::OfflineParaformerModelRknn(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<float> OfflineParaformerModelRknn::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

int32_t OfflineParaformerModelRknn::VocabSize() const {
  return impl_->VocabSize();
}

#if __ANDROID_API__ >= 9
template OfflineParaformerModelRknn::OfflineParaformerModelRknn(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineParaformerModelRknn::OfflineParaformerModelRknn(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
