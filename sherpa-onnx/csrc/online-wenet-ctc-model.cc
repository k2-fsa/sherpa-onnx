// sherpa-onnx/csrc/online-paraformer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-wenet-ctc-model.h"

#include <algorithm>
#include <cmath>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OnlineWenetCtcModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.wenet_ctc.model);
      Init(buf.data(), buf.size());
    }
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.wenet_ctc.model);
      Init(buf.data(), buf.size());
    }
  }
#endif

  std::vector<Ort::Value> Forward(Ort::Value x, Ort::Value offset,
                                  std::vector<Ort::Value> states) {
    // std::array<Ort::Value, 2> inputs = {std::move(features),
    //                                     std::move(features_length)};
    //
    // return sess_->Run({}, input_names_ptr_.data(), inputs.data(),
    // inputs.size(),
    //                   output_names_ptr_.data(), output_names_ptr_.size());
  }

  int32_t VocabSize() const { return vocab_size_; }

  OrtAllocator *Allocator() const { return allocator_; }

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
      PrintModelMetadata(os, meta_data);
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(head_, "head");
    SHERPA_ONNX_READ_META_DATA(num_blocks_, "num_blocks");
    SHERPA_ONNX_READ_META_DATA(output_size_, "output_size");
    SHERPA_ONNX_READ_META_DATA(cnn_module_kernel_, "cnn_module_kernel");
    SHERPA_ONNX_READ_META_DATA(right_context_, "right_context");
    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
  }

 private:
  OnlineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t head_;
  int32_t num_blocks_;
  int32_t output_size_;
  int32_t cnn_module_kernel_;
  int32_t right_context_;
  int32_t subsampling_factor_;
  int32_t vocab_size_;
};

OnlineWenetCtcModel::OnlineWenetCtcModel(const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OnlineWenetCtcModel::OnlineWenetCtcModel(AAssetManager *mgr,
                                         const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OnlineWenetCtcModel::~OnlineWenetCtcModel() = default;

std::vector<Ort::Value> OnlineWenetCtcModel::Forward(
    Ort::Value x, Ort::Value offset, std::vector<Ort::Value> states) const {
  return impl_->Forward(std::move(x), std::move(offset), std::move(states));
}

int32_t OnlineWenetCtcModel::VocabSize() const { return impl_->VocabSize(); }

OrtAllocator *OnlineWenetCtcModel::Allocator() const {
  return impl_->Allocator();
}

}  // namespace sherpa_onnx
