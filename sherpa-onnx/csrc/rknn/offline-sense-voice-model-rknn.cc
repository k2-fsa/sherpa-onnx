// sherpa-onnx/csrc/rknn/offline-sense-voice-model-rknn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/offline-sense-voice-model-rknn.h"

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
#include "sherpa-onnx/csrc/rknn/macros.h"
#include "sherpa-onnx/csrc/rknn/utils.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelRknn::Impl {
 public:
  ~Impl() {
    auto ret = rknn_destroy(ctx_);
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE("Failed to destroy the context");
    }
  }

  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(config_.sense_voice.model);
      Init(buf.data(), buf.size());
    }

    SetCoreMask(ctx_, config_.num_threads);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(mgr, config_.sense_voice.model);
      Init(buf.data(), buf.size());
    }

    SetCoreMask(ctx_, config_.num_threads);
  }

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) {
    features = ApplyLFR(std::move(features));

    std::vector<rknn_input> inputs(input_attrs_.size());

    std::array<int32_t, 4> prompt{language, 1, 2, text_norm};

    inputs[0].index = input_attrs_[0].index;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = input_attrs_[0].fmt;
    inputs[0].buf = reinterpret_cast<void *>(features.data());
    inputs[0].size = features.size() * sizeof(float);

    inputs[1].index = input_attrs_[1].index;
    inputs[1].type = RKNN_TENSOR_INT32;
    inputs[1].fmt = input_attrs_[1].fmt;
    inputs[1].buf = reinterpret_cast<void *>(prompt.data());
    inputs[1].size = prompt.size() * sizeof(int32_t);

    std::vector<float> out(output_attrs_[0].n_elems);

    std::vector<rknn_output> outputs(output_attrs_.size());
    outputs[0].index = output_attrs_[0].index;
    outputs[0].is_prealloc = 1;
    outputs[0].want_float = 1;
    outputs[0].size = out.size() * sizeof(float);
    outputs[0].buf = reinterpret_cast<void *>(out.data());

    rknn_context ctx = 0;
    auto ret = rknn_dup_context(&ctx_, &ctx);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to duplicate the ctx");

    ret = rknn_inputs_set(ctx, inputs.size(), inputs.data());
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set inputs");

    ret = rknn_run(ctx, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run the model");

    ret = rknn_outputs_get(ctx, outputs.size(), outputs.data(), nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get model output");

    rknn_destroy(ctx);

    return out;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &ctx_);

    InitInputOutputAttrs(ctx_, config_.debug, &input_attrs_, &output_attrs_);

    rknn_custom_string custom_string = GetCustomString(ctx_, config_.debug);

    auto meta = Parse(custom_string, config_.debug);

#define SHERPA_ONNX_RKNN_READ_META_DATA_INT(dst, src_key)                     \
  do {                                                                        \
    if (!meta.count(#src_key)) {                                              \
      SHERPA_ONNX_LOGE("'%s' does not exist in the custom_string", #src_key); \
      SHERPA_ONNX_EXIT(-1);                                                   \
    }                                                                         \
                                                                              \
    dst = atoi(meta.at(#src_key).c_str());                                    \
  } while (0)

    SHERPA_ONNX_RKNN_READ_META_DATA_INT(meta_data_.with_itn_id, with_itn);
    SHERPA_ONNX_RKNN_READ_META_DATA_INT(meta_data_.without_itn_id, without_itn);
    SHERPA_ONNX_RKNN_READ_META_DATA_INT(meta_data_.window_size,
                                        lfr_window_size);
    SHERPA_ONNX_RKNN_READ_META_DATA_INT(meta_data_.window_shift,
                                        lfr_window_shift);
    SHERPA_ONNX_RKNN_READ_META_DATA_INT(meta_data_.vocab_size, vocab_size);
    SHERPA_ONNX_RKNN_READ_META_DATA_INT(meta_data_.normalize_samples,
                                        normalize_samples);

    int32_t lang_auto = 0;
    int32_t lang_zh = 0;
    int32_t lang_en = 0;
    int32_t lang_ja = 0;
    int32_t lang_ko = 0;
    int32_t lang_yue = 0;

    SHERPA_ONNX_RKNN_READ_META_DATA_INT(lang_auto, lang_auto);
    SHERPA_ONNX_RKNN_READ_META_DATA_INT(lang_zh, lang_zh);
    SHERPA_ONNX_RKNN_READ_META_DATA_INT(lang_en, lang_en);
    SHERPA_ONNX_RKNN_READ_META_DATA_INT(lang_ja, lang_ja);
    SHERPA_ONNX_RKNN_READ_META_DATA_INT(lang_ko, lang_ko);
    SHERPA_ONNX_RKNN_READ_META_DATA_INT(lang_yue, lang_yue);

    meta_data_.lang2id = {
        {"auto", lang_auto}, {"zh", lang_zh}, {"en", lang_en},
        {"ja", lang_ja},     {"ko", lang_ko}, {"yue", lang_yue},
    };

    // for rknn models, neg_mean and inv_stddev are stored inside the model

#undef SHERPA_ONNX_RKNN_READ_META_DATA_INT

    num_input_frames_ = input_attrs_[0].dims[1];
  }

  std::vector<float> ApplyLFR(std::vector<float> in) const {
    int32_t lfr_window_size = meta_data_.window_size;
    int32_t lfr_window_shift = meta_data_.window_shift;
    int32_t in_feat_dim = 80;

    int32_t in_num_frames = in.size() / in_feat_dim;
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

 private:
  OfflineModelConfig config_;

  rknn_context ctx_ = 0;

  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;

  OfflineSenseVoiceModelMetaData meta_data_;
  int32_t num_input_frames_ = -1;
};

OfflineSenseVoiceModelRknn::~OfflineSenseVoiceModelRknn() = default;

OfflineSenseVoiceModelRknn::OfflineSenseVoiceModelRknn(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSenseVoiceModelRknn::OfflineSenseVoiceModelRknn(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<float> OfflineSenseVoiceModelRknn::Run(std::vector<float> features,
                                                   int32_t language,
                                                   int32_t text_norm) const {
  return impl_->Run(std::move(features), language, text_norm);
}

const OfflineSenseVoiceModelMetaData &
OfflineSenseVoiceModelRknn::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

#if __ANDROID_API__ >= 9
template OfflineSenseVoiceModelRknn::OfflineSenseVoiceModelRknn(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineSenseVoiceModelRknn::OfflineSenseVoiceModelRknn(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
