// sherpa-onnx/csrc/offline-tts-matcha-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-matcha-model.h"

#include <algorithm>
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

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

class OfflineTtsMatchaModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config.matcha.acoustic_model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.matcha.acoustic_model);
    Init(buf.data(), buf.size());
  }

  const OfflineTtsMatchaModelMetaData &GetMetaData() const {
    return meta_data_;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---matcha model---\n";
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
    SHERPA_ONNX_READ_META_DATA(meta_data_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.add_blank, "add_blank",
                                            0);

    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.version, "version", 1);
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_speakers, "n_speakers");
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

  OfflineTtsMatchaModelMetaData meta_data_;
};

OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsMatchaModel::~OfflineTtsMatchaModel() = default;

const OfflineTtsMatchaModelMetaData &OfflineTtsMatchaModel::GetMetaData()
    const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsMatchaModel::OfflineTtsMatchaModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
