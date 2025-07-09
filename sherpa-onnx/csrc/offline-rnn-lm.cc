// sherpa-onnx/csrc/offline-rnn-lm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-rnn-lm.h"

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

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineRnnLM::Impl {
 public:
  explicit Impl(const OfflineLMConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_{GetSessionOptions(config)},
        allocator_{} {
    auto buf = ReadFile(config_.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineLMConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_{GetSessionOptions(config)},
        allocator_{} {
    auto buf = ReadFile(mgr, config_.model);
    Init(buf.data(), buf.size());
  }

  Ort::Value Rescore(Ort::Value x, Ort::Value x_lens) {
    std::array<Ort::Value, 2> inputs = {std::move(x), std::move(x_lens)};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    return std::move(out[0]);
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
  }

 private:
  OfflineLMConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineRnnLM::OfflineRnnLM(const OfflineLMConfig &config)
    : impl_(std::make_unique<Impl>(config)), OfflineLM(config) {}

template <typename Manager>
OfflineRnnLM::OfflineRnnLM(Manager *mgr, const OfflineLMConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)), OfflineLM(config) {}

OfflineRnnLM::~OfflineRnnLM() = default;

Ort::Value OfflineRnnLM::Rescore(Ort::Value x, Ort::Value x_lens) {
  return impl_->Rescore(std::move(x), std::move(x_lens));
}

#if __ANDROID_API__ >= 9
template OfflineRnnLM::OfflineRnnLM(AAssetManager *mgr,
                                    const OfflineLMConfig &config);
#endif

#if __OHOS__
template OfflineRnnLM::OfflineRnnLM(NativeResourceManager *mgr,
                                    const OfflineLMConfig &config);
#endif

}  // namespace sherpa_onnx
