// sherpa-onnx/csrc/offline-tts-kitten-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-kitten-model.h"

#include <algorithm>
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

class OfflineTtsKittenModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto model_buf = ReadFile(config.kitten.model);
    auto voices_buf = ReadFile(config.kitten.voices);
    Init(model_buf.data(), model_buf.size(), voices_buf.data(),
         voices_buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto model_buf = ReadFile(mgr, config.kitten.model);
    auto voices_buf = ReadFile(mgr, config.kitten.voices);
    Init(model_buf.data(), model_buf.size(), voices_buf.data(),
         voices_buf.size());
  }

  const OfflineTtsKittenModelMetaData &GetMetaData() const {
    return meta_data_;
  }

  Ort::Value Run(Ort::Value x, int32_t sid, float speed) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> x_shape = x.GetTensorTypeAndShapeInfo().GetShape();
    if (x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Support only batch_size == 1. Given: %d",
                       static_cast<int32_t>(x_shape[0]));
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t num_speakers = meta_data_.num_speakers;
    int32_t dim1 = style_dim_[1];

    /*const*/ float *p = styles_.data() + sid * dim1;

    std::array<int64_t, 2> style_embedding_shape = {1, dim1};
    Ort::Value style_embedding = Ort::Value::CreateTensor(
        memory_info, p, dim1, style_embedding_shape.data(),
        style_embedding_shape.size());

    int64_t speed_shape = 1;
    if (config_.kitten.length_scale != 1 && speed == 1) {
      speed = 1. / config_.kitten.length_scale;
    }

    Ort::Value speed_tensor =
        Ort::Value::CreateTensor(memory_info, &speed, 1, &speed_shape, 1);

    std::array<Ort::Value, 3> inputs = {
        std::move(x), std::move(style_embedding), std::move(speed_tensor)};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    return std::move(out[0]);
  }

 private:
  void Init(void *model_data, size_t model_data_length, const char *voices_data,
            size_t voices_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---kitten model---\n";
      PrintModelMetadata(os, meta_data);

      os << "----------input names----------\n";
      int32_t i = 0;
      for (const auto &s : input_names_) {
        os << i << " " << s << "\n";
        ++i;
      }
      os << "----------output names----------\n";
      i = 0;
      for (const auto &s : output_names_) {
        os << i << " " << s << "\n";
        ++i;
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR(model_type, "model_type");
    if (model_type != "kitten-tts") {
      SHERPA_ONNX_LOGE(
          "Please download the kitten tts model from us containing meta data");
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(meta_data_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.version, "version", 1);
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_speakers, "n_speakers");
    SHERPA_ONNX_READ_META_DATA(meta_data_.has_espeak, "has_espeak");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.voice, "voice",
                                                "en-us");
    if (meta_data_.has_espeak != 1) {
      SHERPA_ONNX_LOGE("It should require espeak-ng");
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      std::vector<std::string> speaker_names;
      SHERPA_ONNX_READ_META_DATA_VEC_STRING(speaker_names, "speaker_names");
      std::ostringstream os;
      os << "\n";
      for (int32_t i = 0; i != speaker_names.size(); ++i) {
        os << i << "->" << speaker_names[i] << ", ";
      }
      os << "\n";

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    SHERPA_ONNX_READ_META_DATA_VEC(style_dim_, "style_dim");
    if (style_dim_.size() != 2) {
      SHERPA_ONNX_LOGE("style_dim should be 2-d, given: %d",
                       static_cast<int32_t>(style_dim_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (style_dim_[0] != 1) {
      SHERPA_ONNX_LOGE("style_dim[0] should be 1, given: %d", style_dim_[0]);
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t actual_num_floats = voices_data_length / sizeof(float);
    int32_t expected_num_floats =
        style_dim_[0] * style_dim_[1] * meta_data_.num_speakers;

    if (actual_num_floats != expected_num_floats) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Corrupted --kitten-voices '%{public}s'. Expected #floats: "
          "%{public}d, actual: %{public}d",
          config_.kitten.voices.c_str(), expected_num_floats,
          actual_num_floats);
#else
      SHERPA_ONNX_LOGE(
          "Corrupted --kitten-voices '%s'. Expected #floats: %d, actual: %d",
          config_.kitten.voices.c_str(), expected_num_floats,
          actual_num_floats);
#endif

      SHERPA_ONNX_EXIT(-1);
    }

    styles_ = std::vector<float>(
        reinterpret_cast<const float *>(voices_data),
        reinterpret_cast<const float *>(voices_data) + expected_num_floats);
  }

 private:
  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  OfflineTtsKittenModelMetaData meta_data_;
  std::vector<int32_t> style_dim_;

  // (num_speakers, style_dim_[1])
  std::vector<float> styles_;
};

OfflineTtsKittenModel::OfflineTtsKittenModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsKittenModel::OfflineTtsKittenModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsKittenModel::~OfflineTtsKittenModel() = default;

const OfflineTtsKittenModelMetaData &OfflineTtsKittenModel::GetMetaData()
    const {
  return impl_->GetMetaData();
}

Ort::Value OfflineTtsKittenModel::Run(Ort::Value x, int64_t sid /*= 0*/,
                                      float speed /*= 1.0*/) const {
  return impl_->Run(std::move(x), sid, speed);
}

#if __ANDROID_API__ >= 9
template OfflineTtsKittenModel::OfflineTtsKittenModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsKittenModel::OfflineTtsKittenModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
