// sherpa-onnx/csrc/offline-source-separation-spleeter-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-source-separation-spleeter-model.h"

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

class OfflineSourceSeparationSpleeterModel::Impl {
 public:
  explicit Impl(const OfflineSourceSeparationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.spleeter.vocals);
      InitVocals(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.spleeter.accompaniment);
      InitAccompaniment(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineSourceSeparationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.spleeter.vocals);
      InitVocals(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.spleeter.accompaniment);
      InitAccompaniment(buf.data(), buf.size());
    }
  }

  const OfflineSourceSeparationSpleeterModelMetaData &GetMetaData() const {
    return meta_;
  }

  Ort::Value RunVocals(Ort::Value x) const {
    auto out = vocals_sess_->Run({}, vocals_input_names_ptr_.data(), &x, 1,
                                 vocals_output_names_ptr_.data(),
                                 vocals_output_names_ptr_.size());
    return std::move(out[0]);
  }

  Ort::Value RunAccompaniment(Ort::Value x) const {
    auto out =
        accompaniment_sess_->Run({}, accompaniment_input_names_ptr_.data(), &x,
                                 1, accompaniment_output_names_ptr_.data(),
                                 accompaniment_output_names_ptr_.size());
    return std::move(out[0]);
  }

 private:
  void InitVocals(void *model_data, size_t model_data_length) {
    vocals_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(vocals_sess_.get(), &vocals_input_names_,
                  &vocals_input_names_ptr_);

    GetOutputNames(vocals_sess_.get(), &vocals_output_names_,
                   &vocals_output_names_ptr_);

    Ort::ModelMetadata meta_data = vocals_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---vocals model---\n";
      PrintModelMetadata(os, meta_data);

      os << "----------input names----------\n";
      int32_t i = 0;
      for (const auto &s : vocals_input_names_) {
        os << i << " " << s << "\n";
        ++i;
      }
      os << "----------output names----------\n";
      i = 0;
      for (const auto &s : vocals_output_names_) {
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
    if (model_type != "spleeter") {
      SHERPA_ONNX_LOGE("Expect model type 'spleeter'. Given: '%s'",
                       model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(meta_.num_stems, "stems");
    if (meta_.num_stems != 2) {
      SHERPA_ONNX_LOGE("Only 2stems is supported. Given %d stems",
                       meta_.num_stems);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void InitAccompaniment(void *model_data, size_t model_data_length) {
    accompaniment_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(accompaniment_sess_.get(), &accompaniment_input_names_,
                  &accompaniment_input_names_ptr_);

    GetOutputNames(accompaniment_sess_.get(), &accompaniment_output_names_,
                   &accompaniment_output_names_ptr_);
  }

 private:
  OfflineSourceSeparationModelConfig config_;
  OfflineSourceSeparationSpleeterModelMetaData meta_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> vocals_sess_;

  std::vector<std::string> vocals_input_names_;
  std::vector<const char *> vocals_input_names_ptr_;

  std::vector<std::string> vocals_output_names_;
  std::vector<const char *> vocals_output_names_ptr_;

  std::unique_ptr<Ort::Session> accompaniment_sess_;

  std::vector<std::string> accompaniment_input_names_;
  std::vector<const char *> accompaniment_input_names_ptr_;

  std::vector<std::string> accompaniment_output_names_;
  std::vector<const char *> accompaniment_output_names_ptr_;
};

OfflineSourceSeparationSpleeterModel::~OfflineSourceSeparationSpleeterModel() =
    default;

OfflineSourceSeparationSpleeterModel::OfflineSourceSeparationSpleeterModel(
    const OfflineSourceSeparationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSourceSeparationSpleeterModel::OfflineSourceSeparationSpleeterModel(
    Manager *mgr, const OfflineSourceSeparationModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

Ort::Value OfflineSourceSeparationSpleeterModel::RunVocals(Ort::Value x) const {
  return impl_->RunVocals(std::move(x));
}

Ort::Value OfflineSourceSeparationSpleeterModel::RunAccompaniment(
    Ort::Value x) const {
  return impl_->RunAccompaniment(std::move(x));
}

const OfflineSourceSeparationSpleeterModelMetaData &
OfflineSourceSeparationSpleeterModel::GetMetaData() const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineSourceSeparationSpleeterModel::
    OfflineSourceSeparationSpleeterModel(
        AAssetManager *mgr, const OfflineSourceSeparationModelConfig &config);
#endif

#if __OHOS__
template OfflineSourceSeparationSpleeterModel::
    OfflineSourceSeparationSpleeterModel(
        NativeResourceManager *mgr,
        const OfflineSourceSeparationModelConfig &config);
#endif

}  // namespace sherpa_onnx
