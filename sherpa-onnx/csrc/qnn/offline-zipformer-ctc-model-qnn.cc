// sherpa-onnx/csrc/qnn/offline-zipformer-ctc-model-qnn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/offline-zipformer-ctc-model-qnn.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>  // NOLINT
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
#include "sherpa-onnx/csrc/qnn/macros.h"
#include "sherpa-onnx/csrc/qnn/qnn-backend.h"
#include "sherpa-onnx/csrc/qnn/qnn-model.h"

namespace sherpa_onnx {

class OfflineZipformerCtcModelQnn::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    backend_ = std::make_unique<QnnBackend>(
        config.zipformer_ctc.qnn_config.backend_lib, config_.debug);

    const auto &context_binary =
        config_.zipformer_ctc.qnn_config.context_binary;

    if (context_binary.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init from model lib since context binary is not given");
      }

      InitFromModelLib();

      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Skip generating context binary since you don't provide a path to "
            "save it");
      }
    } else if (!FileExists(context_binary)) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init from model lib since context binary '%s' does not exist",
            context_binary.c_str());
      }

      InitFromModelLib();

      CreateContextBinary();
    } else {
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Init from context binary '%s'",
                         context_binary.c_str());
      }
      InitFromContextBinary();
    }

    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    SHERPA_ONNX_LOGE(
        "Please copy all files from assets to SD card and set assetManager to "
        "null");
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<float> Run(std::vector<float> features) {
    int32_t num_frames = features.size() / feat_dim_;

    if (num_frames != max_num_frames_) {
      if (num_frames > max_num_frames_) {
        SHERPA_ONNX_LOGE(
            "Number of input frames %d is too large. Truncate it to %d frames.",
            num_frames, max_num_frames_);

        SHERPA_ONNX_LOGE(
            "Recognition result may be truncated/incomplete. Please select a "
            "model accepting longer audios.");
      }

      features.resize(max_num_frames_ * feat_dim_);

      num_frames = max_num_frames_;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    model_->SetInputTensorData("x", features.data(), features.size());

    model_->Run();

    return model_->GetOutputTensorData("log_probs");
  }

  int32_t VocabSize() const { return vocab_size_; }
  int32_t SubsamplingFactor() const { return subsampling_factor_; }

 private:
  void InitFromModelLib() {
    backend_->InitContext();

    model_ = std::make_unique<QnnModel>(config_.zipformer_ctc.model,
                                        backend_.get(), config_.debug);
  }

  void InitFromContextBinary() {
    model_ = std::make_unique<QnnModel>(
        config_.zipformer_ctc.qnn_config.context_binary,
        config_.zipformer_ctc.qnn_config.system_lib, backend_.get(),
        BinaryContextTag{}, config_.debug);
  }

  void CreateContextBinary() {
    const auto &context_binary =
        config_.zipformer_ctc.qnn_config.context_binary;

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Creating context binary '%s'.", context_binary.c_str());
    }

    bool ok = model_->SaveBinaryContext(context_binary);

    if (!ok) {
      SHERPA_ONNX_LOGE("Failed to save context binary to '%s'",
                       context_binary.c_str());
    }

    if (config_.debug && ok) {
      SHERPA_ONNX_LOGE("Saved context binary to '%s'.", context_binary.c_str());
      SHERPA_ONNX_LOGE(
          "It should be super fast the next time you init the system.");
      SHERPA_ONNX_LOGE("Remember to also provide libQnnSystem.so.");
    }
  }

  void PostInit() { CheckModel(); }

  void CheckModel() {
    const auto &input_tensor_names = model_->InputTensorNames();
    if (input_tensor_names.size() != 1) {
      SHERPA_ONNX_LOGE("Expect 1 input tensor. Actual %d",
                       static_cast<int32_t>(input_tensor_names.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (input_tensor_names[0] != "x") {
      SHERPA_ONNX_LOGE("The 1st input should be x, actual '%s'",
                       input_tensor_names[0].c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int32_t> x_shape = model_->TensorShape(input_tensor_names[0]);
    if (x_shape.size() != 3) {
      SHERPA_ONNX_LOGE("The 1st input should be 3-d, actual '%d'",
                       static_cast<int32_t>(x_shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The x.shape[0] should be 1, actual '%d'", x_shape[0]);
      SHERPA_ONNX_EXIT(-1);
    }

    max_num_frames_ = x_shape[1];
    feat_dim_ = x_shape[2];

    if (!model_->HasTensor("log_probs")) {
      SHERPA_ONNX_LOGE("Model does not have output node 'log_probs'");
      SHERPA_ONNX_EXIT(-1);
    }

    auto out_shape = model_->TensorShape("log_probs");
    vocab_size_ = out_shape[2];

    subsampling_factor_ = max_num_frames_ / out_shape[1];
    if (config_.debug) {
      SHERPA_ONNX_LOGE("max_num_frames: %d", max_num_frames_);
      SHERPA_ONNX_LOGE("feat_dim: %d", feat_dim_);
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
      SHERPA_ONNX_LOGE("subsampling_factor: %d", subsampling_factor_);
    }
  }

 private:
  std::mutex mutex_;

  OfflineModelConfig config_;

  std::unique_ptr<QnnBackend> backend_;
  std::unique_ptr<QnnModel> model_;

  int32_t max_num_frames_ = 0;
  int32_t feat_dim_ = 0;
  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 1;
};

OfflineZipformerCtcModelQnn::OfflineZipformerCtcModelQnn(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineZipformerCtcModelQnn::OfflineZipformerCtcModelQnn(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineZipformerCtcModelQnn::~OfflineZipformerCtcModelQnn() = default;

std::vector<float> OfflineZipformerCtcModelQnn::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

int32_t OfflineZipformerCtcModelQnn::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OfflineZipformerCtcModelQnn::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

#if __ANDROID_API__ >= 9
template OfflineZipformerCtcModelQnn::OfflineZipformerCtcModelQnn(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineZipformerCtcModelQnn::OfflineZipformerCtcModelQnn(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
