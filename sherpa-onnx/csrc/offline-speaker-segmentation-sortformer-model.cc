// sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model.cc
//
// Copyright (c)  2026  Silvio Tomatis

#include "sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model.h"

#include <array>
#include <cmath>
#include <cstdlib>
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

// Read a float
#define SHERPA_ONNX_READ_META_DATA_FLOAT(dst, src_key)                     \
  do {                                                                     \
    auto value = LookupCustomModelMetaData(meta_data, src_key, allocator); \
    if (value.empty()) {                                                   \
      SHERPA_ONNX_LOGE("'%s' does not exist in the metadata", src_key);    \
      SHERPA_ONNX_EXIT(-1);                                                \
    }                                                                      \
                                                                           \
    dst = std::strtof(value.c_str(), nullptr);                             \
  } while (0)

namespace sherpa_onnx {

class OfflineSpeakerSegmentationSortformerModel::Impl {
 public:
  explicit Impl(const OfflineSpeakerSegmentationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.sortformer.model), sess_opts_);
    Init(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineSpeakerSegmentationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.sortformer.model);
    Init(buf.data(), buf.size());
  }

  const OfflineSpeakerSegmentationSortformerModelMetaData &GetModelMetaData()
      const {
    return meta_data_;
  }

  OrtAllocator *Allocator() { return allocator_; }

  std::pair<Ort::Value, Ort::Value> Forward(Ort::Value features,
                                            Ort::Value cached_embeds,
                                            int32_t num_frames) {
    const auto features_shape = features.GetTensorTypeAndShapeInfo().GetShape();
    const auto cached_shape =
        cached_embeds.GetTensorTypeAndShapeInfo().GetShape();
    int64_t num_embeds =
        features_shape[1] / meta_data_.cache.subsampling_factor;
    int64_t num_output_frames =
        (cached_shape[1] + num_embeds) * meta_data_.cache.subsampling_factor;
    auto length = Ort::Value::CreateTensor<int64_t>(allocator_, nullptr, 0);
    *length.GetTensorMutableData<int64_t>() = num_frames;
    std::array<Ort::Value, 3> inputs = {
        std::move(features), std::move(cached_embeds), std::move(length)};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    if (out[0].GetTensorTypeAndShapeInfo().GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        out[1].GetTensorTypeAndShapeInfo().GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        out[0].GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>{1, num_output_frames,
                                 meta_data_.cache.num_speakers} ||
        out[1].GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>{1, num_embeds, meta_data_.cache.hidden_size}) {
      SHERPA_ONNX_LOGE("Invalid Sortformer output shapes/types");
      SHERPA_ONNX_EXIT(-1);
    }

    return {std::move(out[0]), std::move(out[1])};
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    if (model_data) {
      sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                             model_data_length, sess_opts_);
    } else if (!sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass model data or initialize the session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR(model_type, "model_type");
    if (model_type != "nemotron3_diarization") {
      SHERPA_ONNX_LOGE("Unsupported Sortformer model type: '%s'",
                       model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t version;
    SHERPA_ONNX_READ_META_DATA(version, "version");
    if (version != 2 ||
        input_names_ != std::vector<std::string>{"features", "cached_embeds",
                                                 "num_frames"} ||
        output_names_ != std::vector<std::string>{"probs", "chunk_embeds"}) {
      SHERPA_ONNX_LOGE("Unsupported Sortformer model interface/version");
      SHERPA_ONNX_EXIT(-1);
    }

    auto &m = meta_data_;
    SHERPA_ONNX_READ_META_DATA(m.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA(m.n_fft, "n_fft");
    SHERPA_ONNX_READ_META_DATA(m.win_length, "win_length");
    SHERPA_ONNX_READ_META_DATA(m.hop_length, "hop_length");
    SHERPA_ONNX_READ_META_DATA(m.num_mel_bins, "num_mel_bins");
    SHERPA_ONNX_READ_META_DATA_FLOAT(m.preemphasis, "preemphasis");

    SHERPA_ONNX_READ_META_DATA(m.chunk_length, "chunk_length");
    SHERPA_ONNX_READ_META_DATA(m.chunk_right_context, "chunk_right_context");

    auto &c = m.cache;
    SHERPA_ONNX_READ_META_DATA(c.num_speakers, "num_speakers");
    SHERPA_ONNX_READ_META_DATA(c.hidden_size, "hidden_size");
    SHERPA_ONNX_READ_META_DATA(c.subsampling_factor, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA(c.fifo_length, "fifo_length");
    SHERPA_ONNX_READ_META_DATA(c.speaker_cache_update_period,
                               "speaker_cache_update_period");
    SHERPA_ONNX_READ_META_DATA(c.speaker_cache_length, "speaker_cache_length");
    SHERPA_ONNX_READ_META_DATA(c.num_silence_frames,
                               "speaker_cache_silence_frames_per_speaker");
    SHERPA_ONNX_READ_META_DATA_FLOAT(c.prediction_score_threshold,
                                     "prediction_score_threshold");
    SHERPA_ONNX_READ_META_DATA_FLOAT(c.latest_frames_score_boost,
                                     "latest_frames_score_boost");
    SHERPA_ONNX_READ_META_DATA_FLOAT(c.min_positive_scores_rate,
                                     "min_positive_scores_rate");
    SHERPA_ONNX_READ_META_DATA_FLOAT(c.strong_boost_rate, "strong_boost_rate");
    SHERPA_ONNX_READ_META_DATA_FLOAT(c.weak_boost_rate, "weak_boost_rate");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(c.silence_embeds, "silence_embeds");

    if (static_cast<int32_t>(c.silence_embeds.size()) != c.hidden_size) {
      SHERPA_ONNX_LOGE("Expect %d values for silence_embeds. Given: %d",
                       c.hidden_size,
                       static_cast<int32_t>(c.silence_embeds.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (m.sample_rate < 1 || m.n_fft < 1 || m.win_length < 2 ||
        m.win_length > m.n_fft || m.hop_length < 1 || m.num_mel_bins < 1 ||
        !std::isfinite(m.preemphasis) || m.chunk_length < 1 ||
        m.chunk_right_context < 0 || c.hidden_size < 1 ||
        c.subsampling_factor < 1 || c.num_speakers < 1 || c.fifo_length < 0 ||
        c.num_silence_frames < 0 || c.speaker_cache_update_period < 1 ||
        c.speaker_cache_length <
            (1LL + c.num_silence_frames) * c.num_speakers ||
        !(c.prediction_score_threshold > 0 &&
          c.prediction_score_threshold < 1) ||
        !std::isfinite(c.latest_frames_score_boost) ||
        !(c.min_positive_scores_rate >= 0 && c.min_positive_scores_rate <= 1) ||
        !(c.strong_boost_rate >= 0 && std::isfinite(c.strong_boost_rate)) ||
        !(c.weak_boost_rate >= 0 && std::isfinite(c.weak_boost_rate))) {
      SHERPA_ONNX_LOGE("Invalid Sortformer model meta data");
      SHERPA_ONNX_EXIT(-1);
    }
  }

 private:
  OfflineSpeakerSegmentationModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  OfflineSpeakerSegmentationSortformerModelMetaData meta_data_;
};

OfflineSpeakerSegmentationSortformerModel::
    OfflineSpeakerSegmentationSortformerModel(  // NOLINT
        const OfflineSpeakerSegmentationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}  // NOLINT

template <typename Manager>
OfflineSpeakerSegmentationSortformerModel::
    OfflineSpeakerSegmentationSortformerModel(  // NOLINT
        Manager *mgr, const OfflineSpeakerSegmentationModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}  // NOLINT

OfflineSpeakerSegmentationSortformerModel::
    ~OfflineSpeakerSegmentationSortformerModel() = default;  // NOLINT

const OfflineSpeakerSegmentationSortformerModelMetaData &
OfflineSpeakerSegmentationSortformerModel::GetModelMetaData() const {
  return impl_->GetModelMetaData();
}

OrtAllocator *OfflineSpeakerSegmentationSortformerModel::Allocator() const {
  return impl_->Allocator();
}

std::pair<Ort::Value, Ort::Value>
OfflineSpeakerSegmentationSortformerModel::Forward(Ort::Value features,
                                                   Ort::Value cached_embeds,
                                                   int32_t num_frames) const {
  return impl_->Forward(std::move(features), std::move(cached_embeds),
                        num_frames);
}

#if __ANDROID_API__ >= 9
template OfflineSpeakerSegmentationSortformerModel::
    OfflineSpeakerSegmentationSortformerModel(  // NOLINT
        AAssetManager *mgr,
        const OfflineSpeakerSegmentationModelConfig &config);
#endif

#if __OHOS__
template OfflineSpeakerSegmentationSortformerModel::
    OfflineSpeakerSegmentationSortformerModel(  // NOLINT
        NativeResourceManager *mgr,
        const OfflineSpeakerSegmentationModelConfig &config);
#endif

}  // namespace sherpa_onnx
