// sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-model.h"

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

class OfflineSpeechDenoiserGtcrnModel::Impl {
 public:
  explicit Impl(const OfflineSpeechDenoiserModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.gtcrn.model);
      Init(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineSpeechDenoiserModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.gtcrn.model);
      Init(buf.data(), buf.size());
    }
  }

  const OfflineSpeechDenoiserGtcrnModelMetaData &GetMetaData() const {
    return meta_;
  }

  States GetInitStates() {
    Ort::Value conv_cache = Ort::Value::CreateTensor<float>(
        allocator_, meta_.conv_cache_shape.data(),
        meta_.conv_cache_shape.size());

    Ort::Value tra_cache = Ort::Value::CreateTensor<float>(
        allocator_, meta_.tra_cache_shape.data(), meta_.tra_cache_shape.size());

    Ort::Value inter_cache = Ort::Value::CreateTensor<float>(
        allocator_, meta_.inter_cache_shape.data(),
        meta_.inter_cache_shape.size());

    Fill<float>(&conv_cache, 0);
    Fill<float>(&tra_cache, 0);
    Fill<float>(&inter_cache, 0);

    std::vector<Ort::Value> states;

    states.reserve(3);
    states.push_back(std::move(conv_cache));
    states.push_back(std::move(tra_cache));
    states.push_back(std::move(inter_cache));

    return states;
  }

  std::pair<Ort::Value, States> Run(Ort::Value x, States states) const {
    std::vector<Ort::Value> inputs;
    inputs.reserve(1 + states.size());
    inputs.push_back(std::move(x));
    for (auto &s : states) {
      inputs.push_back(std::move(s));
    }

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    std::vector<Ort::Value> next_states;
    next_states.reserve(out.size() - 1);
    for (int32_t k = 1; k < out.size(); ++k) {
      next_states.push_back(std::move(out[k]));
    }

    return {std::move(out[0]), std::move(next_states)};
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
      os << "---gtcrn model---\n";
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
    if (model_type != "gtcrn") {
      SHERPA_ONNX_LOGE("Expect model type 'gtcrn'. Given: '%s'",
                       model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(meta_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA(meta_.n_fft, "n_fft");
    SHERPA_ONNX_READ_META_DATA(meta_.hop_length, "hop_length");
    SHERPA_ONNX_READ_META_DATA(meta_.window_length, "window_length");
    SHERPA_ONNX_READ_META_DATA_STR(meta_.window_type, "window_type");
    SHERPA_ONNX_READ_META_DATA(meta_.version, "version");

    SHERPA_ONNX_READ_META_DATA_VEC(meta_.conv_cache_shape, "conv_cache_shape");
    SHERPA_ONNX_READ_META_DATA_VEC(meta_.tra_cache_shape, "tra_cache_shape");
    SHERPA_ONNX_READ_META_DATA_VEC(meta_.inter_cache_shape,
                                   "inter_cache_shape");
  }

 private:
  OfflineSpeechDenoiserModelConfig config_;
  OfflineSpeechDenoiserGtcrnModelMetaData meta_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineSpeechDenoiserGtcrnModel::~OfflineSpeechDenoiserGtcrnModel() = default;

OfflineSpeechDenoiserGtcrnModel::OfflineSpeechDenoiserGtcrnModel(
    const OfflineSpeechDenoiserModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSpeechDenoiserGtcrnModel::OfflineSpeechDenoiserGtcrnModel(
    Manager *mgr, const OfflineSpeechDenoiserModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineSpeechDenoiserGtcrnModel::States
OfflineSpeechDenoiserGtcrnModel::GetInitStates() const {
  return impl_->GetInitStates();
}

std::pair<Ort::Value, OfflineSpeechDenoiserGtcrnModel::States>
OfflineSpeechDenoiserGtcrnModel::Run(Ort::Value x, States states) const {
  return impl_->Run(std::move(x), std::move(states));
}

const OfflineSpeechDenoiserGtcrnModelMetaData &
OfflineSpeechDenoiserGtcrnModel::GetMetaData() const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineSpeechDenoiserGtcrnModel::OfflineSpeechDenoiserGtcrnModel(
    AAssetManager *mgr, const OfflineSpeechDenoiserModelConfig &config);
#endif

#if __OHOS__
template OfflineSpeechDenoiserGtcrnModel::OfflineSpeechDenoiserGtcrnModel(
    NativeResourceManager *mgr, const OfflineSpeechDenoiserModelConfig &config);
#endif

}  // namespace sherpa_onnx
