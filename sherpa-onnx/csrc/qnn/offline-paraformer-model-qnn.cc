// sherpa-onnx/csrc/qnn/offline-paraformer-model-qnn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/offline-paraformer-model-qnn.h"

#include <algorithm>
#include <array>
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
#include "sherpa-onnx/csrc/qnn/macros.h"
#include "sherpa-onnx/csrc/qnn/qnn-backend.h"
#include "sherpa-onnx/csrc/qnn/qnn-model.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineParaformerModelQnn::Impl {
 public:
  Impl(const OfflineModelConfig &config) : config_(config) {
    std::vector<std::string> filenames;
    SplitStringToVector(config_.paraformer.model, ",", false, &filenames);
    if (filenames.size() != 3) {
      SHERPA_ONNX_LOGE("Invalid Paraformer QNN model '%s'",
                       config_.paraformer.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<std::string> binary_filenames;
    SplitStringToVector(config_.paraformer.qnn_config.context_binary, ",",
                        false, &binary_filenames);

    bool ok = InitEncoder(filenames[0],
                          binary_filenames.empty() ? "" : binary_filenames[0]);
    if (!ok) {
      SHERPA_ONNX_LOGE("Failed to init encoder with '%s'",
                       filenames[0].c_str());
      return;
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) {}

  std::vector<float> Run(std::vector<float> features) const { return {}; }

  int32_t VocabSize() const { return 0; }

 private:
  bool InitEncoder(const std::string &lib_filename,
                   const std::string &context_binary) {
    encoder_backend_ = std::make_unique<QnnBackend>(
        config_.paraformer.qnn_config.backend_lib, config_.debug);

    if (context_binary.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init from encoder model lib since context binary is not given");
      }

      InitEncoderFromModelLib(lib_filename);

      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Skip generating encoder context binary since you don't provide a "
            "path to "
            "save it");
      }
    } else if (!FileExists(context_binary)) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init encoder from model lib since context binary '%s' does not "
            "exist",
            context_binary.c_str());
      }

      InitEncoderFromModelLib(lib_filename);
      // CreateEncoderContextBinary();
    }

    PostInitEncoder();

    return true;
  }

  void InitEncoderFromModelLib(const std::string &lib_filename) {
    encoder_backend_->InitContext();
    encoder_model_ = std::make_unique<QnnModel>(
        lib_filename, encoder_backend_.get(), config_.debug);
  }

  void PostInitEncoder() { CheckEncoderModel(); }

  void CheckEncoderModel() {
    const auto &input_tensor_names = encoder_model_->InputTensorNames();
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

    std::vector<int32_t> x_shape =
        encoder_model_->TensorShape(input_tensor_names[0]);
    if (x_shape.size() != 3) {
      SHERPA_ONNX_LOGE("The 1st input should be 3-d, actual '%d'",
                       static_cast<int32_t>(x_shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The x.shape[0] should be 1, actual '%d'", x_shape[0]);
      SHERPA_ONNX_EXIT(-1);
    }

    feat_dim_ = x_shape[2];

    if (!encoder_model_->HasTensor("encoder_out")) {
      SHERPA_ONNX_LOGE("Model does not have output node 'encoder_out'");
      SHERPA_ONNX_EXIT(-1);
    }

    expected_num_frames_ = x_shape[1];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("expected_num_frames_: %d", expected_num_frames_);
      SHERPA_ONNX_LOGE("feat_dim: %d", feat_dim_);
    }
  }

 private:
  std::mutex mutex_;
  OfflineModelConfig config_;

  std::unique_ptr<QnnBackend> encoder_backend_;
  std::unique_ptr<QnnModel> encoder_model_;
  int32_t expected_num_frames_ = 0;
  int32_t feat_dim_ = 0;
};

OfflineParaformerModelQnn::~OfflineParaformerModelQnn() = default;

OfflineParaformerModelQnn::OfflineParaformerModelQnn(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineParaformerModelQnn::OfflineParaformerModelQnn(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<float> OfflineParaformerModelQnn::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

int32_t OfflineParaformerModelQnn::VocabSize() const {
  return impl_->VocabSize();
}

#if __ANDROID_API__ >= 9
template OfflineParaformerModelQnn::OfflineParaformerModelQnn(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineParaformerModelQnn::OfflineParaformerModelQnn(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
