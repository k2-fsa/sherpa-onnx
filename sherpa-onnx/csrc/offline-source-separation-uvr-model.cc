// sherpa-onnx/csrc/offline-source-separation-uvr-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-source-separation-uvr-model.h"

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
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineSourceSeparationUvrModel::Impl {
 public:
  explicit Impl(const OfflineSourceSeparationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config.uvr.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineSourceSeparationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.uvr.model);
    Init(buf.data(), buf.size());
  }

  const OfflineSourceSeparationUvrModelMetaData &GetMetaData() const {
    return meta_;
  }

  Ort::Value Run(Ort::Value x) const {
    auto out = sess_->Run({}, input_names_ptr_.data(), &x, 1,
                          output_names_ptr_.data(), output_names_ptr_.size());
    return std::move(out[0]);
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---UVR model---\n";
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
    if (model_type != "UVR") {
      SHERPA_ONNX_LOGE("Expect model type 'UVR'. Given: '%s'",
                       model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(meta_.num_stems, "stems");
    if (meta_.num_stems != 2) {
      SHERPA_ONNX_LOGE("Only 2stems is supported. Given %d stems",
                       meta_.num_stems);
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(meta_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA(meta_.n_fft, "n_fft");
    SHERPA_ONNX_READ_META_DATA(meta_.center, "center");
    SHERPA_ONNX_READ_META_DATA(meta_.window_length, "win_length");
    SHERPA_ONNX_READ_META_DATA(meta_.hop_length, "hop_length");
    SHERPA_ONNX_READ_META_DATA(meta_.dim_t, "dim_t");
    SHERPA_ONNX_READ_META_DATA(meta_.dim_f, "dim_f");
    SHERPA_ONNX_READ_META_DATA(meta_.dim_c, "dim_c");
    SHERPA_ONNX_READ_META_DATA_STR(meta_.window_type, "window_type");

    meta_.margin = meta_.sample_rate;
  }

 private:
  OfflineSourceSeparationModelConfig config_;
  OfflineSourceSeparationUvrModelMetaData meta_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineSourceSeparationUvrModel::~OfflineSourceSeparationUvrModel() = default;

OfflineSourceSeparationUvrModel::OfflineSourceSeparationUvrModel(
    const OfflineSourceSeparationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSourceSeparationUvrModel::OfflineSourceSeparationUvrModel(
    Manager *mgr, const OfflineSourceSeparationModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

Ort::Value OfflineSourceSeparationUvrModel::Run(Ort::Value x) const {
  return impl_->Run(std::move(x));
}

const OfflineSourceSeparationUvrModelMetaData &
OfflineSourceSeparationUvrModel::GetMetaData() const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineSourceSeparationUvrModel::OfflineSourceSeparationUvrModel(
    AAssetManager *mgr, const OfflineSourceSeparationModelConfig &config);
#endif

#if __OHOS__
template OfflineSourceSeparationUvrModel::OfflineSourceSeparationUvrModel(
    NativeResourceManager *mgr,
    const OfflineSourceSeparationModelConfig &config);
#endif

}  // namespace sherpa_onnx
