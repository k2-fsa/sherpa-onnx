// sherpa-onnx/csrc/qnn/offline-paraformer-model-qnn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/offline-paraformer-model-qnn.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
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
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/qnn/macros.h"
#include "sherpa-onnx/csrc/qnn/qnn-backend.h"
#include "sherpa-onnx/csrc/qnn/qnn-model.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineParaformerModelQnn::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    std::vector<std::string> filenames;
    SplitStringToVector(config_.paraformer.model, ",", true, &filenames);
    if (!filenames.empty()) {
      if (filenames.size() != 3) {
        SHERPA_ONNX_LOGE("Invalid Paraformer QNN model '%s'",
                         config_.paraformer.model.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    std::vector<std::string> binary_filenames;
    SplitStringToVector(config_.paraformer.qnn_config.context_binary, ",", true,
                        &binary_filenames);
    if (!binary_filenames.empty()) {
      if (binary_filenames.size() != 3) {
        SHERPA_ONNX_LOGE(
            "There should be 3 files for Paraformer context binary. Actual: "
            "%d. '%s'",
            static_cast<int32_t>(binary_filenames.size()),
            config_.paraformer.qnn_config.context_binary.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    if (filenames.empty() && binary_filenames.empty()) {
      SHERPA_ONNX_LOGE(
          "You need to provide either a model or a context binary for "
          "Paraformer with QNN");
      SHERPA_ONNX_EXIT(-1);
    }

    bool ok = InitEncoder(filenames.empty() ? "" : filenames[0],
                          binary_filenames.empty() ? "" : binary_filenames[0]);
    if (!ok) {
      SHERPA_ONNX_LOGE(
          "Failed to init encoder with lib file '%s', context binary: '%s'",
          filenames.empty() ? "" : filenames[0].c_str(),
          binary_filenames.empty() ? "" : binary_filenames[0].c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    ok = InitPredictor(filenames.empty() ? "" : filenames[1],
                       binary_filenames.empty() ? "" : binary_filenames[1]);
    if (!ok) {
      SHERPA_ONNX_LOGE(
          "Failed to init predictor with lib file '%s', context binary: '%s'",
          filenames.empty() ? "" : filenames[1].c_str(),
          binary_filenames.empty() ? "" : binary_filenames[1].c_str());
      return;
    }

    ok = InitDecoder(filenames.empty() ? "" : filenames[2],
                     binary_filenames.empty() ? "" : binary_filenames[2]);
    if (!ok) {
      SHERPA_ONNX_LOGE(
          "Failed to init decoder with lib file '%s', context binary: '%s'",
          filenames.empty() ? "" : filenames[2].c_str(),
          binary_filenames.empty() ? "" : binary_filenames[2].c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) {
    SHERPA_ONNX_LOGE("TODO(fangjun): To be implemented");
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<float> Run(std::vector<float> features) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<float> encoder_out = RunEncoder(std::move(features));
    std::vector<float> transposed_encoder_out =
        Transpose(encoder_out.data(), encoder_out_dim1_, encoder_out_dim2_);

    std::vector<float> alphas = RunPredictor(transposed_encoder_out);

    std::vector<float> acoustic_embedding =
        ComputeAcousticEmbedding(encoder_out, alphas, encoder_out_dim2_);

    int32_t num_tokens = acoustic_embedding.size() / encoder_out_dim2_;

    acoustic_embedding.resize(encoder_out.size());

    std::vector<float> transposed_acoustic_embedding = Transpose(
        acoustic_embedding.data(), encoder_out_dim1_, encoder_out_dim2_);

    std::vector<float> decoder_out = RunDecoder(
        transposed_encoder_out, transposed_acoustic_embedding, num_tokens);

    decoder_out = Transpose(decoder_out.data(), vocab_size_, encoder_out_dim1_);
    decoder_out.resize(num_tokens * vocab_size_);
    return decoder_out;
  }

  int32_t VocabSize() const { return vocab_size_; }

 private:
  std::vector<float> RunEncoder(std::vector<float> features) const {
    features = ApplyLFR(std::move(features));
    if (features.empty()) {
      return {};
    }

    encoder_model_->SetInputTensorData("x", features.data(), features.size());
    encoder_model_->Run();
    return encoder_model_->GetOutputTensorData("encoder_out");
  }

  std::vector<float> RunPredictor(
      const std::vector<float> &transposed_encoder_out) const {
    predictor_model_->SetInputTensorData("encoder_out",
                                         transposed_encoder_out.data(),
                                         transposed_encoder_out.size());
    predictor_model_->Run();
    return predictor_model_->GetOutputTensorData("alphas");
  }

  std::vector<float> RunDecoder(
      const std::vector<float> &transposed_encoder_out,
      const std::vector<float> &transposed_acoustic_embedding,
      int32_t num_tokens) const {
    std::vector<int32_t> mask(encoder_out_dim1_, 1);
    std::fill(mask.begin() + num_tokens, mask.end(), 0);

    decoder_model_->SetInputTensorData("encoder_out",
                                       transposed_encoder_out.data(),
                                       transposed_encoder_out.size());

    decoder_model_->SetInputTensorData("acoustic_embedding",
                                       transposed_acoustic_embedding.data(),
                                       transposed_acoustic_embedding.size());

    decoder_model_->SetInputTensorData("mask", mask.data(), mask.size());

    decoder_model_->Run();

    return decoder_model_->GetOutputTensorData("decoder_out");
  }

  std::vector<float> ApplyLFR(std::vector<float> in) const {
    int32_t lfr_window_size = 7;
    int32_t lfr_window_shift = 6;
    int32_t in_feat_dim = 80;

    int32_t in_num_frames = in.size() / in_feat_dim;
    if (in_num_frames < lfr_window_size) {
      return {};
    }

    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;

    if (out_num_frames > num_input_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          out_num_frames, num_input_frames_);

      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios.");

      out_num_frames = num_input_frames_;
    }

    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(num_input_frames_ * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  bool InitEncoder(const std::string &lib_filename,
                   const std::string &context_binary) {
    encoder_backend_ = std::make_unique<QnnBackend>(
        config_.paraformer.qnn_config.backend_lib, config_.debug);

    if (context_binary.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init from encoder model lib '%s' since context binary is not "
            "given.",
            lib_filename.c_str());
      }

      InitEncoderFromModelLib(lib_filename);

      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Skip generating encoder context binary since you don't provide a "
            "path to save it");
      }
    } else if (!FileExists(context_binary)) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init encoder from model lib '%s' since context binary '%s' does "
            "not exist",
            lib_filename.c_str(), context_binary.c_str());
      }

      InitEncoderFromModelLib(lib_filename);

      CreateContextBinary(encoder_model_.get(), context_binary);
    } else {
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Init from encoder context binary '%s'",
                         context_binary.c_str());
      }
      InitEncoderFromContextBinary(context_binary);
    }

    PostInitEncoder();

    return true;
  }

  bool InitPredictor(const std::string &lib_filename,
                     const std::string &context_binary) {
    predictor_backend_ = std::make_unique<QnnBackend>(
        config_.paraformer.qnn_config.backend_lib, config_.debug);

    if (context_binary.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init from predictor model lib '%s' since context binary is not "
            "given.",
            lib_filename.c_str());
      }

      InitPredictorFromModelLib(lib_filename);

      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Skip generating predictor context binary since you don't provide "
            "a path to save it");
      }
    } else if (!FileExists(context_binary)) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init predictor from model lib '%s' since context binary '%s' does "
            "not exist",
            lib_filename.c_str(), context_binary.c_str());
      }

      InitPredictorFromModelLib(lib_filename);
      CreateContextBinary(predictor_model_.get(), context_binary);
    } else {
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Init from predictor context binary '%s'",
                         context_binary.c_str());
      }
      InitPredictorFromContextBinary(context_binary);
    }

    PostInitPredictor();

    return true;
  }

  bool InitDecoder(const std::string &lib_filename,
                   const std::string &context_binary) {
    decoder_backend_ = std::make_unique<QnnBackend>(
        config_.paraformer.qnn_config.backend_lib, config_.debug);

    if (context_binary.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init from decoder model lib since context binary is not given");
      }

      InitDecoderFromModelLib(lib_filename);

      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Skip generating decoder context binary since you don't provide "
            "a path to save it");
      }
    } else if (!FileExists(context_binary)) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init decoder from model lib since context binary '%s' does not "
            "exist",
            context_binary.c_str());
      }

      InitDecoderFromModelLib(lib_filename);
      CreateContextBinary(decoder_model_.get(), context_binary);
    } else {
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Init from decoder context binary '%s'",
                         context_binary.c_str());
      }
      InitDecoderFromContextBinary(context_binary);
    }

    PostInitDecoder();

    return true;
  }

  void InitEncoderFromModelLib(const std::string &lib_filename) {
    encoder_backend_->InitContext();
    encoder_model_ = std::make_unique<QnnModel>(
        lib_filename, encoder_backend_.get(), config_.debug);
  }

  void InitPredictorFromModelLib(const std::string &lib_filename) {
    predictor_backend_->InitContext();
    predictor_model_ = std::make_unique<QnnModel>(
        lib_filename, predictor_backend_.get(), config_.debug);
  }

  void InitDecoderFromModelLib(const std::string &lib_filename) {
    decoder_backend_->InitContext();
    decoder_model_ = std::make_unique<QnnModel>(
        lib_filename, decoder_backend_.get(), config_.debug);
  }

  void CreateContextBinary(QnnModel *model, const std::string &context_binary) {
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Creating context binary '%s'.", context_binary.c_str());
    }

    bool ok = model->SaveBinaryContext(context_binary);

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

  void InitEncoderFromContextBinary(const std::string &context_binary) {
    if (config_.paraformer.qnn_config.system_lib.empty()) {
      SHERPA_ONNX_LOGE(
          "You should provide --paraformer.qnn-system-lib if you also provide "
          "context binary");
      SHERPA_ONNX_EXIT(-1);
    }

    encoder_model_ = std::make_unique<QnnModel>(
        context_binary, config_.paraformer.qnn_config.system_lib,
        encoder_backend_.get(), BinaryContextTag{}, config_.debug);
  }

  void InitPredictorFromContextBinary(const std::string &context_binary) {
    if (config_.paraformer.qnn_config.system_lib.empty()) {
      SHERPA_ONNX_LOGE(
          "You should provide --paraformer.qnn-system-lib if you also provide "
          "context binary");
      SHERPA_ONNX_EXIT(-1);
    }

    predictor_model_ = std::make_unique<QnnModel>(
        context_binary, config_.paraformer.qnn_config.system_lib,
        predictor_backend_.get(), BinaryContextTag{}, config_.debug);
  }

  void InitDecoderFromContextBinary(const std::string &context_binary) {
    if (config_.paraformer.qnn_config.system_lib.empty()) {
      SHERPA_ONNX_LOGE(
          "You should provide --paraformer.qnn-system-lib if you also provide "
          "context binary");
      SHERPA_ONNX_EXIT(-1);
    }

    decoder_model_ = std::make_unique<QnnModel>(
        context_binary, config_.paraformer.qnn_config.system_lib,
        decoder_backend_.get(), BinaryContextTag{}, config_.debug);
  }

  void PostInitEncoder() { CheckEncoderModel(); }

  void PostInitPredictor() { CheckPredictorModel(); }

  void PostInitDecoder() { CheckDecoderModel(); }

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

    num_input_frames_ = x_shape[1];
    feat_dim_ = x_shape[2];

    if (!encoder_model_->HasTensor("encoder_out")) {
      SHERPA_ONNX_LOGE("Model does not have output node 'encoder_out'");
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int32_t> encoder_out_shape =
        encoder_model_->TensorShape("encoder_out");

    encoder_out_dim1_ = encoder_out_shape[1];
    encoder_out_dim2_ = encoder_out_shape[2];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("num_input_frames: %d", num_input_frames_);
      SHERPA_ONNX_LOGE("feat_dim: %d", feat_dim_);
      SHERPA_ONNX_LOGE("encoder_out_dim1: %d", encoder_out_dim1_);
      SHERPA_ONNX_LOGE("encoder_out_dim2: %d", encoder_out_dim2_);
    }
  }

  void CheckPredictorModel() {
    const auto &input_tensor_names = predictor_model_->InputTensorNames();
    if (input_tensor_names.size() != 1) {
      SHERPA_ONNX_LOGE("Expect 1 input tensor. Actual %d",
                       static_cast<int32_t>(input_tensor_names.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (input_tensor_names[0] != "encoder_out") {
      SHERPA_ONNX_LOGE("The 1st input should be encoder_out, actual '%s'",
                       input_tensor_names[0].c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int32_t> x_shape =
        predictor_model_->TensorShape(input_tensor_names[0]);
    if (x_shape.size() != 3) {
      SHERPA_ONNX_LOGE("The 1st input should be 3-d, actual '%d'",
                       static_cast<int32_t>(x_shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The x.shape[0] should be 1, actual '%d'", x_shape[0]);
      SHERPA_ONNX_EXIT(-1);
    }

    if (x_shape[1] != encoder_out_dim2_) {
      SHERPA_ONNX_LOGE(
          "The input dim 1 of the predictor should be %d, given: %d",
          encoder_out_dim2_, x_shape[1]);
      SHERPA_ONNX_EXIT(-1);
    }

    if (x_shape[2] != encoder_out_dim1_) {
      SHERPA_ONNX_LOGE(
          "The input dim 2 of the predictor should be %d, given: %d",
          encoder_out_dim1_, x_shape[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    if (!predictor_model_->HasTensor("alphas")) {
      SHERPA_ONNX_LOGE("Model does not have output node 'alphas'");
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int32_t> alphas_shape = predictor_model_->TensorShape("alphas");
    if (alphas_shape.size() != 2) {
      SHERPA_ONNX_LOGE("alphas should be 2-d, given: %d",
                       static_cast<int32_t>(alphas_shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (alphas_shape[0] != 1) {
      SHERPA_ONNX_LOGE("We support only batch size 1 for alphas. Given: %d",
                       alphas_shape[0]);
      SHERPA_ONNX_EXIT(-1);
    }

    if (alphas_shape[1] != encoder_out_dim1_) {
      SHERPA_ONNX_LOGE("Expected output dim %d for alphas. Given: %d",
                       encoder_out_dim1_, alphas_shape[1]);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void CheckDecoderModel() {
    const auto &input_tensor_names = decoder_model_->InputTensorNames();
    if (input_tensor_names.size() != 3) {
      SHERPA_ONNX_LOGE("Expect 3 input tensors. Actual %d",
                       static_cast<int32_t>(input_tensor_names.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (input_tensor_names[0] != "encoder_out") {
      SHERPA_ONNX_LOGE("The 1st input should be encoder_out, actual '%s'",
                       input_tensor_names[0].c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (input_tensor_names[1] != "acoustic_embedding") {
      SHERPA_ONNX_LOGE(
          "The 2nd input should be acoustic_embedding, actual '%s'",
          input_tensor_names[1].c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (input_tensor_names[2] != "mask") {
      SHERPA_ONNX_LOGE("The 3rd input should be mask, actual '%s'",
                       input_tensor_names[2].c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!decoder_model_->HasTensor("decoder_out")) {
      SHERPA_ONNX_LOGE("Model does not have output node 'decoder_out'");
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int32_t> decoder_out_shape =
        decoder_model_->TensorShape("decoder_out");
    if (decoder_out_shape.size() != 3) {
      SHERPA_ONNX_LOGE("decoder_out should be 3-d, given: %d",
                       static_cast<int32_t>(decoder_out_shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (decoder_out_shape[0] != 1) {
      SHERPA_ONNX_LOGE("We support only batch size 1 for decoder. Given: %d",
                       decoder_out_shape[0]);
      SHERPA_ONNX_EXIT(-1);
    }

    if (decoder_out_shape[2] != encoder_out_dim1_) {
      SHERPA_ONNX_LOGE("Expected output dim %d for decoder_out. Given: %d",
                       encoder_out_dim1_, decoder_out_shape[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    vocab_size_ = decoder_out_shape[1];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
    }
  }

 private:
  std::mutex mutex_;
  OfflineModelConfig config_;

  std::unique_ptr<QnnBackend> encoder_backend_;
  std::unique_ptr<QnnModel> encoder_model_;

  std::unique_ptr<QnnBackend> predictor_backend_;
  std::unique_ptr<QnnModel> predictor_model_;

  std::unique_ptr<QnnBackend> decoder_backend_;
  std::unique_ptr<QnnModel> decoder_model_;

  int32_t num_input_frames_ = 0;
  int32_t feat_dim_ = 0;

  int32_t encoder_out_dim1_ = 0;
  int32_t encoder_out_dim2_ = 0;
  int32_t vocab_size_ = 0;
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
