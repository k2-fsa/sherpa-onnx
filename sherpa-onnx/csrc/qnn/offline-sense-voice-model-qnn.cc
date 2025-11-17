// sherpa-onnx/csrc/qnn/offline-sense-voice-model-qnn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/offline-sense-voice-model-qnn.h"

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

class OfflineSenseVoiceModelQnn::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    backend_ = std::make_unique<QnnBackend>(
        config.sense_voice.qnn_config.backend_lib, config_.debug);

    const auto &context_binary = config_.sense_voice.qnn_config.context_binary;

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

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) {
    std::lock_guard<std::mutex> lock(mutex_);

    features = ApplyLFR(std::move(features));

    int32_t num_frames = features.size() / feat_dim_;

    model_->SetInputTensorData("x", features.data(), features.size());

    std::array<int32_t, 4> prompt = {language, 1, 2, text_norm};
    model_->SetInputTensorData("prompt", prompt.data(), prompt.size());

    model_->Run();

    return model_->GetOutputTensorData("logits");
  }

 private:
  void InitFromModelLib() {
    backend_->InitContext();

    model_ = std::make_unique<QnnModel>(config_.sense_voice.model,
                                        backend_.get(), config_.debug);
  }

  void InitFromContextBinary() {
    model_ = std::make_unique<QnnModel>(
        config_.sense_voice.qnn_config.context_binary,
        config_.sense_voice.qnn_config.system_lib, backend_.get(),
        BinaryContextTag{}, config_.debug);
  }

  void CreateContextBinary() {
    const auto &context_binary = config_.sense_voice.qnn_config.context_binary;

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
    if (input_tensor_names.size() != 2) {
      SHERPA_ONNX_LOGE("Expect two input tensors. Actual %d",
                       static_cast<int32_t>(input_tensor_names.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (input_tensor_names[0] != "x") {
      SHERPA_ONNX_LOGE("The 1st input should be x, actual '%s'",
                       input_tensor_names[0].c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (input_tensor_names[1] != "prompt") {
      SHERPA_ONNX_LOGE("The 2nd input should be prompt, actual '%s'",
                       input_tensor_names[1].c_str());
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

    if (x_shape[2] != feat_dim_) {
      SHERPA_ONNX_LOGE("The x.shape[2] should be %d, actual '%d'", feat_dim_,
                       x_shape[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int32_t> prompt_shape =
        model_->TensorShape(input_tensor_names[1]);

    if (prompt_shape.size() != 1) {
      SHERPA_ONNX_LOGE("The 2nd input should be 1-d, actual '%d'",
                       static_cast<int32_t>(prompt_shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (prompt_shape[0] != 4) {
      SHERPA_ONNX_LOGE("The prompt.shape[0] should be 4, actual '%d'",
                       prompt_shape[0]);
      SHERPA_ONNX_EXIT(-1);
    }

    if (!model_->HasTensor("logits")) {
      SHERPA_ONNX_LOGE("Model does not have output node 'logits'");
      SHERPA_ONNX_EXIT(-1);
    }

    expected_num_frames_ = x_shape[1];
  }

  std::vector<float> ApplyLFR(std::vector<float> in) const {
    int32_t lfr_window_size = meta_data_.window_size;
    int32_t lfr_window_shift = meta_data_.window_shift;
    int32_t in_feat_dim = 80;

    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;

    if (out_num_frames > expected_num_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          out_num_frames, expected_num_frames_);

      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios.");

      out_num_frames = expected_num_frames_;
    }

    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    // if out_num_frames < expected_num_frames_, it uses 0 padding
    std::vector<float> out(expected_num_frames_ * out_feat_dim, 0);

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
  std::mutex mutex_;

  OfflineModelConfig config_;
  OfflineSenseVoiceModelMetaData meta_data_;

  std::unique_ptr<QnnBackend> backend_;
  std::unique_ptr<QnnModel> model_;

  int32_t expected_num_frames_ = 0;
  int32_t feat_dim_ = 560;
};

OfflineSenseVoiceModelQnn::OfflineSenseVoiceModelQnn(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSenseVoiceModelQnn::OfflineSenseVoiceModelQnn(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineSenseVoiceModelQnn::~OfflineSenseVoiceModelQnn() = default;

std::vector<float> OfflineSenseVoiceModelQnn::Run(std::vector<float> features,
                                                  int32_t language,
                                                  int32_t text_norm) const {
  return impl_->Run(std::move(features), language, text_norm);
}

const OfflineSenseVoiceModelMetaData &
OfflineSenseVoiceModelQnn::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

#if __ANDROID_API__ >= 9
template OfflineSenseVoiceModelQnn::OfflineSenseVoiceModelQnn(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineSenseVoiceModelQnn::OfflineSenseVoiceModelQnn(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
