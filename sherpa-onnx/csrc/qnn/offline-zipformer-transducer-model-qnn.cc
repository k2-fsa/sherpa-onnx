// sherpa-onnx/csrc/qnn/offline-zipformer-transducer-model-qnn.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/offline-zipformer-transducer-model-qnn.h"

#include <algorithm>
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
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/qnn/macros.h"
#include "sherpa-onnx/csrc/qnn/qnn-backend.h"
#include "sherpa-onnx/csrc/qnn/qnn-model.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineZipformerTransducerModelQnn::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) { Init(); }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    (void)mgr;
    SHERPA_ONNX_LOGE(
        "Please copy all files from assets to SD card and set assetManager to "
        "null");
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<float> RunEncoder(std::vector<float> features) const {
    int32_t num_frames = static_cast<int32_t>(features.size()) / feat_dim_;
    if (num_frames == 0) {
      return {};
    }

    if (num_frames > max_num_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          num_frames, max_num_frames_);
      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios.");
      num_frames = max_num_frames_;
      features.resize(static_cast<size_t>(num_frames) * feat_dim_);
    }

    features.resize(static_cast<size_t>(max_num_frames_) * feat_dim_);
    features = Transpose(features.data(), max_num_frames_, feat_dim_);

    std::lock_guard<std::mutex> lock(mutex_);
    encoder_->SetInputTensorData(encoder_input_name_, features.data(),
                                 static_cast<int32_t>(features.size()));
    encoder_->Run();

    std::vector<float> encoder_out =
        encoder_->GetOutputTensorData(encoder_output_name_);
    encoder_out.resize(static_cast<size_t>(NumEncoderFrames(num_frames)) *
                       encoder_out_dim_);
    return encoder_out;
  }

  std::vector<float> RunDecoder(const std::vector<int32_t> &tokens) const {
    if (static_cast<int32_t>(tokens.size()) != context_size_) {
      SHERPA_ONNX_LOGE("Expected decoder input size %d, given %d",
                       context_size_, static_cast<int32_t>(tokens.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    decoder_->SetInputTensorData(decoder_input_name_, tokens.data(),
                                 static_cast<int32_t>(tokens.size()));
    decoder_->Run();
    return decoder_->GetOutputTensorData(decoder_output_name_);
  }

  std::vector<float> RunJoiner(const float *encoder_out,
                               const std::vector<float> &decoder_out) const {
    if (static_cast<int32_t>(decoder_out.size()) != decoder_out_dim_) {
      SHERPA_ONNX_LOGE("Expected joiner decoder input size %d, given %d",
                       decoder_out_dim_,
                       static_cast<int32_t>(decoder_out.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    joiner_->SetInputTensorData(joiner_encoder_input_name_, encoder_out,
                                encoder_out_dim_);
    joiner_->SetInputTensorData(joiner_decoder_input_name_, decoder_out.data(),
                                static_cast<int32_t>(decoder_out.size()));
    joiner_->Run();
    return joiner_->GetOutputTensorData(joiner_output_name_);
  }

  int32_t ContextSize() const { return context_size_; }
  int32_t VocabSize() const { return vocab_size_; }
  int32_t SubsamplingFactor() const { return subsampling_factor_; }
  int32_t EncoderDim() const { return encoder_out_dim_; }

  int32_t NumEncoderFrames(int32_t num_frames) const {
    if (num_frames <= 0) {
      return 0;
    }

    num_frames = std::min(num_frames, max_num_frames_);
    int32_t ans = (num_frames + subsampling_factor_ - 1) / subsampling_factor_;
    return std::min(ans, max_num_encoder_frames_);
  }

 private:
  void Init() {
    ParseContextBinaries();
    encoder_backend_ = std::make_unique<QnnBackend>(
        config_.transducer.qnn_config.backend_lib, config_.debug);
    decoder_backend_ = std::make_unique<QnnBackend>(
        config_.transducer.qnn_config.backend_lib, config_.debug);
    joiner_backend_ = std::make_unique<QnnBackend>(
        config_.transducer.qnn_config.backend_lib, config_.debug);

    InitEncoder();
    InitDecoder();
    InitJoiner();

    CheckEncoder();
    CheckDecoder();
    CheckJoiner();
  }

  void ParseContextBinaries() {
    SplitStringToVector(config_.transducer.qnn_config.context_binary, ",", true,
                        &context_binaries_);
    if (!context_binaries_.empty() && context_binaries_.size() != 3) {
      SHERPA_ONNX_LOGE(
          "There should be 3 files for offline transducer context binary. "
          "Actual: %d. '%s'",
          static_cast<int32_t>(context_binaries_.size()),
          config_.transducer.qnn_config.context_binary.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void CreateContextBinary(QnnModel *model,
                           const std::string &context_binary) const {
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
      SHERPA_ONNX_LOGE("Remember to also provide libQnnSystem.so.");
    }
  }

  void InitModelFromLib(const std::string &filename, QnnBackend *backend,
                        std::unique_ptr<QnnModel> *model) const {
    backend->InitContext();
    *model = std::make_unique<QnnModel>(filename, backend, config_.debug);
  }

  void InitModelFromContextBinary(const std::string &filename,
                                  QnnBackend *backend,
                                  std::unique_ptr<QnnModel> *model) const {
    if (config_.transducer.qnn_config.system_lib.empty()) {
      SHERPA_ONNX_LOGE(
          "You should provide --transducer.qnn-system-lib if you also provide "
          "context binary");
      SHERPA_ONNX_EXIT(-1);
    }

    *model = std::make_unique<QnnModel>(
        filename, config_.transducer.qnn_config.system_lib, backend,
        BinaryContextTag{}, config_.debug);
  }

  void InitComponent(const std::string &lib_filename,
                     const std::string &context_binary, const char *name,
                     QnnBackend *backend,
                     std::unique_ptr<QnnModel> *model) const {
    if (context_binary.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init from %s model lib '%s' since context binary is not given.",
            name, lib_filename.c_str());
      }

      InitModelFromLib(lib_filename, backend, model);
      return;
    }

    if (!FileExists(context_binary)) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init %s from model lib '%s' since context binary '%s' does not "
            "exist",
            name, lib_filename.c_str(), context_binary.c_str());
      }

      InitModelFromLib(lib_filename, backend, model);
      CreateContextBinary(model->get(), context_binary);
    } else {
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Init from %s context binary '%s'", name,
                         context_binary.c_str());
      }

      InitModelFromContextBinary(context_binary, backend, model);
    }
  }

  void InitEncoder() {
    InitComponent(config_.transducer.encoder_filename,
                  context_binaries_.empty() ? "" : context_binaries_[0],
                  "encoder", encoder_backend_.get(), &encoder_);
  }

  void InitDecoder() {
    InitComponent(config_.transducer.decoder_filename,
                  context_binaries_.empty() ? "" : context_binaries_[1],
                  "decoder", decoder_backend_.get(), &decoder_);
  }

  void InitJoiner() {
    InitComponent(config_.transducer.joiner_filename,
                  context_binaries_.empty() ? "" : context_binaries_[2],
                  "joiner", joiner_backend_.get(), &joiner_);
  }

  void CheckEncoder() {
    encoder_input_name_ = "x";
    encoder_output_name_ = "encoder_out";

    if (!encoder_->HasTensor(encoder_input_name_)) {
      SHERPA_ONNX_LOGE("The encoder model does not have input '%s'",
                       encoder_input_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!encoder_->HasTensor(encoder_output_name_)) {
      SHERPA_ONNX_LOGE("The encoder model does not have output '%s'",
                       encoder_output_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    auto x_shape = encoder_->TensorShape(encoder_input_name_);
    if (x_shape.size() != 3 || x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The encoder input x should be of shape [1, ?, ?]");
      SHERPA_ONNX_EXIT(-1);
    }

    feat_dim_ = x_shape[1];
    max_num_frames_ = x_shape[2];

    auto out_shape = encoder_->TensorShape(encoder_output_name_);
    if (out_shape.size() != 3 || out_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The encoder output should be of shape [1, ?, ?]");
      SHERPA_ONNX_EXIT(-1);
    }

    max_num_encoder_frames_ = out_shape[1];
    encoder_out_dim_ = out_shape[2];

    if (feat_dim_ <= 0 || max_num_frames_ <= 0 || max_num_encoder_frames_ <= 0 ||
        encoder_out_dim_ <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid encoder shapes. feat_dim: %d, max_num_frames: %d, "
          "max_num_encoder_frames: %d, encoder_out_dim: %d",
          feat_dim_, max_num_frames_, max_num_encoder_frames_,
          encoder_out_dim_);
      SHERPA_ONNX_EXIT(-1);
    }

    subsampling_factor_ = max_num_frames_ / max_num_encoder_frames_;
    if (subsampling_factor_ <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid offline transducer QNN encoder shapes. Input frames: %d, "
          "output frames: %d",
          max_num_frames_, max_num_encoder_frames_);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void CheckDecoder() {
    const auto &input_names = decoder_->InputTensorNames();
    if (input_names.size() != 1) {
      SHERPA_ONNX_LOGE("Expected one decoder input. Given %d",
                       static_cast<int32_t>(input_names.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    decoder_input_name_ = input_names[0];
    auto in_shape = decoder_->TensorShape(decoder_input_name_);
    if (in_shape.size() != 2 || in_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Expected decoder input shape [1, context_size]");
      SHERPA_ONNX_EXIT(-1);
    }
    context_size_ = in_shape[1];

    decoder_output_name_ = "decoder_out";
    if (!decoder_->HasTensor(decoder_output_name_)) {
      SHERPA_ONNX_LOGE("The decoder model does not have output '%s'",
                       decoder_output_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    auto out_shape = decoder_->TensorShape(decoder_output_name_);
    if (out_shape.size() != 2 || out_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The decoder output should be of shape [1, decoder_dim]");
      SHERPA_ONNX_EXIT(-1);
    }
    decoder_out_dim_ = out_shape[1];

    if (context_size_ <= 0 || decoder_out_dim_ <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid decoder shapes. context_size: %d, decoder_out_dim: %d",
          context_size_, decoder_out_dim_);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void CheckJoiner() {
    joiner_encoder_input_name_ = "encoder_out";
    joiner_decoder_input_name_ = "decoder_out";
    joiner_output_name_ = "logit";

    if (!joiner_->HasTensor(joiner_encoder_input_name_)) {
      SHERPA_ONNX_LOGE("The joiner model does not have input '%s'",
                       joiner_encoder_input_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!joiner_->HasTensor(joiner_decoder_input_name_)) {
      SHERPA_ONNX_LOGE("The joiner model does not have input '%s'",
                       joiner_decoder_input_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!joiner_->HasTensor(joiner_output_name_)) {
      SHERPA_ONNX_LOGE("The joiner model does not have output '%s'",
                       joiner_output_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    auto encoder_in_shape = joiner_->TensorShape(joiner_encoder_input_name_);
    if (encoder_in_shape.size() != 2 || encoder_in_shape[0] != 1 ||
        encoder_in_shape[1] != encoder_out_dim_) {
      SHERPA_ONNX_LOGE(
          "The joiner input '%s' should be of shape [1, %d]",
          joiner_encoder_input_name_.c_str(), encoder_out_dim_);
      SHERPA_ONNX_EXIT(-1);
    }

    auto decoder_in_shape = joiner_->TensorShape(joiner_decoder_input_name_);
    if (decoder_in_shape.size() != 2 || decoder_in_shape[0] != 1 ||
        decoder_in_shape[1] != decoder_out_dim_) {
      SHERPA_ONNX_LOGE(
          "The joiner input '%s' should be of shape [1, %d]",
          joiner_decoder_input_name_.c_str(), decoder_out_dim_);
      SHERPA_ONNX_EXIT(-1);
    }

    auto out_shape = joiner_->TensorShape(joiner_output_name_);
    if (out_shape.size() != 2 || out_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The joiner output should be of shape [1, vocab_size]");
      SHERPA_ONNX_EXIT(-1);
    }

    vocab_size_ = out_shape[1];
    if (vocab_size_ <= 0) {
      SHERPA_ONNX_LOGE("Invalid joiner output shape. vocab_size: %d",
                       vocab_size_);
      SHERPA_ONNX_EXIT(-1);
    }
  }

 private:
  mutable std::mutex mutex_;

  OfflineModelConfig config_;

  std::unique_ptr<QnnBackend> encoder_backend_;
  std::unique_ptr<QnnBackend> decoder_backend_;
  std::unique_ptr<QnnBackend> joiner_backend_;

  std::unique_ptr<QnnModel> encoder_;
  std::unique_ptr<QnnModel> decoder_;
  std::unique_ptr<QnnModel> joiner_;
  std::vector<std::string> context_binaries_;

  std::string encoder_input_name_;
  std::string encoder_output_name_;
  std::string decoder_input_name_;
  std::string decoder_output_name_;
  std::string joiner_encoder_input_name_;
  std::string joiner_decoder_input_name_;
  std::string joiner_output_name_;

  int32_t max_num_frames_ = 0;
  int32_t feat_dim_ = 0;
  int32_t max_num_encoder_frames_ = 0;
  int32_t encoder_out_dim_ = 0;
  int32_t decoder_out_dim_ = 0;
  int32_t context_size_ = 0;
  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 0;
};

OfflineZipformerTransducerModelQnn::OfflineZipformerTransducerModelQnn(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineZipformerTransducerModelQnn::OfflineZipformerTransducerModelQnn(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineZipformerTransducerModelQnn::~OfflineZipformerTransducerModelQnn() =
    default;

std::vector<float> OfflineZipformerTransducerModelQnn::RunEncoder(
    std::vector<float> features) const {
  return impl_->RunEncoder(std::move(features));
}

std::vector<float> OfflineZipformerTransducerModelQnn::RunDecoder(
    const std::vector<int32_t> &tokens) const {
  return impl_->RunDecoder(tokens);
}

std::vector<float> OfflineZipformerTransducerModelQnn::RunJoiner(
    const float *encoder_out, const std::vector<float> &decoder_out) const {
  return impl_->RunJoiner(encoder_out, decoder_out);
}

int32_t OfflineZipformerTransducerModelQnn::ContextSize() const {
  return impl_->ContextSize();
}

int32_t OfflineZipformerTransducerModelQnn::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OfflineZipformerTransducerModelQnn::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

int32_t OfflineZipformerTransducerModelQnn::EncoderDim() const {
  return impl_->EncoderDim();
}

int32_t OfflineZipformerTransducerModelQnn::NumEncoderFrames(
    int32_t num_frames) const {
  return impl_->NumEncoderFrames(num_frames);
}

#if __ANDROID_API__ >= 9
template OfflineZipformerTransducerModelQnn::OfflineZipformerTransducerModelQnn(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineZipformerTransducerModelQnn::OfflineZipformerTransducerModelQnn(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
