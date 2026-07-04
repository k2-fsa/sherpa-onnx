// sherpa-onnx/csrc/qnn/offline-parakeet-tdt-model-qnn.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/offline-parakeet-tdt-model-qnn.h"

#include <algorithm>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <tuple>
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

class OfflineParakeetTdtModelQnn::Impl {
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

  // Encoder:
  //   Input  "x":           (1, max_num_frames, feat_dim)
  //   Output "encoder_out": (1, num_encoder_frames, encoder_out_dim)
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

    // Pad to max_num_frames_
    features.resize(static_cast<size_t>(max_num_frames_) * feat_dim_);

    // Note: The QNN model expects input shape [1, max_num_frames, feat_dim].
    // Features are already in (num_frames, feat_dim) row-major order.
    // No transpose is needed here (unlike zipformer which needs transpose).

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

  // Decoder (LSTM-based):
  //   Input  "y":            (1, 1)                           int32, a single
  //                                                     token id
  //   Input  "h":            (num_layers, hidden_dim, 1)      float, hidden
  //                                                     state
  //   Input  "c":            (num_layers, hidden_dim, 1)      float, cell
  //                                                     state
  //   Output "decoder_out":  (1, 1, decoder_out_dim)         float
  //   Output "next_h":       (num_layers, 1, hidden_dim)     float, updated
  //                                                     hidden state
  //   Output "next_c":       (num_layers, 1, hidden_dim)     float, updated
  //                                                     cell state
  //
  //   Note: h/c input shape is (num_layers, hidden_dim, 1) while
  //         next_h/next_c output shape is (num_layers, 1, hidden_dim).
  //         Since batch_size=1, no transpose is needed.
  std::vector<std::vector<float>> GetDecoderInitStates() const {
    return {std::vector<float>(h_size_, 0.0f),
            std::vector<float>(c_size_, 0.0f)};
  }

  std::pair<std::vector<float>, std::vector<std::vector<float>>> RunDecoder(
      int32_t token, std::vector<std::vector<float>> states) const {
    std::lock_guard<std::mutex> lock(mutex_);
    decoder_->SetInputTensorData(decoder_input_y_name_, &token, 1);
    decoder_->SetInputTensorData(decoder_input_h_name_, states[0].data(),
                                 static_cast<int32_t>(states[0].size()));
    decoder_->SetInputTensorData(decoder_input_c_name_, states[1].data(),
                                 static_cast<int32_t>(states[1].size()));
    decoder_->Run();

    std::vector<float> decoder_out =
        decoder_->GetOutputTensorData(decoder_output_name_);
    states[0] = decoder_->GetOutputTensorData(decoder_output_h_name_);
    states[1] = decoder_->GetOutputTensorData(decoder_output_c_name_);

    return {std::move(decoder_out), std::move(states)};
  }

  // Joiner:
  //   Input  "encoder_out": (1, encoder_out_dim, 1) float
  //   Input  "decoder_out": (1, decoder_out_dim, 1) float
  //   Output "log_probs":   (1, 1, 1, vocab_size + num_durations) float
  //     Despite the name "log_probs", the output contains raw logits
  //     (log_softmax is NOT applied).
  //     First vocab_size elements are token logits,
  //     remaining num_durations elements are duration logits.
  std::vector<float> RunJoiner(const float *encoder_out,
                               const std::vector<float> &decoder_out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    joiner_->SetInputTensorData(joiner_encoder_input_name_, encoder_out,
                                encoder_out_dim_);
    joiner_->SetInputTensorData(joiner_decoder_input_name_, decoder_out.data(),
                                static_cast<int32_t>(decoder_out.size()));
    joiner_->Run();
    return joiner_->GetOutputTensorData(joiner_output_name_);
  }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }
  int32_t EncoderDim() const { return encoder_out_dim_; }
  int32_t FeatDim() const { return feat_dim_; }

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
          "There should be 3 files for offline parakeet TDT context binary. "
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

    // QNN Parakeet TDT encoder input shape: [1, max_num_frames, feat_dim]
    // (Note: the ONNX model uses [1, feat_dim, max_num_frames] instead)
    max_num_frames_ = x_shape[1];
    feat_dim_ = x_shape[2];

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
          "Invalid parakeet TDT QNN encoder shapes. Input frames: %d, "
          "output frames: %d",
          max_num_frames_, max_num_encoder_frames_);
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("encoder input '%s': [%d, %d, %d]",
                       encoder_input_name_.c_str(), 1, max_num_frames_,
                       feat_dim_);
      SHERPA_ONNX_LOGE("encoder output '%s': [%d, %d, %d]",
                       encoder_output_name_.c_str(), 1, max_num_encoder_frames_,
                       encoder_out_dim_);
      SHERPA_ONNX_LOGE("subsampling_factor: %d", subsampling_factor_);
    }
  }

  void CheckDecoder() {
    // Parakeet TDT decoder has 3 inputs: y (token), h (hidden state),
    // c (cell state) and 3 outputs: decoder_out, next_h, next_c
    decoder_input_y_name_ = "y";
    decoder_input_h_name_ = "h";
    decoder_input_c_name_ = "c";
    decoder_output_name_ = "decoder_out";
    decoder_output_h_name_ = "next_h";
    decoder_output_c_name_ = "next_c";

    if (!decoder_->HasTensor(decoder_input_y_name_)) {
      SHERPA_ONNX_LOGE("The decoder model does not have input '%s'",
                       decoder_input_y_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!decoder_->HasTensor(decoder_input_h_name_)) {
      SHERPA_ONNX_LOGE("The decoder model does not have input '%s'",
                       decoder_input_h_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!decoder_->HasTensor(decoder_input_c_name_)) {
      SHERPA_ONNX_LOGE("The decoder model does not have input '%s'",
                       decoder_input_c_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!decoder_->HasTensor(decoder_output_name_)) {
      SHERPA_ONNX_LOGE("The decoder model does not have output '%s'",
                       decoder_output_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!decoder_->HasTensor(decoder_output_h_name_)) {
      SHERPA_ONNX_LOGE("The decoder model does not have output '%s'",
                       decoder_output_h_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!decoder_->HasTensor(decoder_output_c_name_)) {
      SHERPA_ONNX_LOGE("The decoder model does not have output '%s'",
                       decoder_output_c_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    auto y_shape = decoder_->TensorShape(decoder_input_y_name_);
    if (y_shape.size() != 2 || y_shape[0] != 1 || y_shape[1] != 1) {
      SHERPA_ONNX_LOGE(
          "Expected decoder input 'y' shape [1, 1]. Actual: [%d, %d]",
          y_shape[0], y_shape.size() > 1 ? y_shape[1] : 0);
      SHERPA_ONNX_EXIT(-1);
    }

    // h shape: (num_layers, hidden_dim, 1)
    auto h_shape = decoder_->TensorShape(decoder_input_h_name_);
    if (h_shape.size() != 3 || h_shape[2] != 1) {
      SHERPA_ONNX_LOGE(
          "Expected h shape (num_layers, hidden_dim, 1). Actual: [%d, %d, %d]",
          h_shape.size() > 0 ? h_shape[0] : 0,
          h_shape.size() > 1 ? h_shape[1] : 0,
          h_shape.size() > 2 ? h_shape[2] : 0);
      SHERPA_ONNX_EXIT(-1);
    }
    int32_t num_layers = h_shape[0];
    int32_t hidden_dim = h_shape[1];
    h_size_ = num_layers * hidden_dim * 1;

    // c shape: (num_layers, hidden_dim, 1)
    auto c_shape = decoder_->TensorShape(decoder_input_c_name_);
    if (c_shape.size() != 3 || c_shape[0] != num_layers ||
        c_shape[1] != hidden_dim || c_shape[2] != 1) {
      SHERPA_ONNX_LOGE(
          "Expected c shape (%d, %d, 1). Actual: [%d, %d, %d]", num_layers,
          hidden_dim, c_shape.size() > 0 ? c_shape[0] : 0,
          c_shape.size() > 1 ? c_shape[1] : 0,
          c_shape.size() > 2 ? c_shape[2] : 0);
      SHERPA_ONNX_EXIT(-1);
    }
    c_size_ = num_layers * hidden_dim * 1;

    // next_h shape: (num_layers, 1, hidden_dim)
    auto next_h_shape = decoder_->TensorShape(decoder_output_h_name_);
    if (next_h_shape.size() != 3 || next_h_shape[0] != num_layers ||
        next_h_shape[1] != 1 || next_h_shape[2] != hidden_dim) {
      SHERPA_ONNX_LOGE(
          "Expected next_h shape (%d, 1, %d). Actual: [%d, %d, %d]",
          num_layers, hidden_dim,
          next_h_shape.size() > 0 ? next_h_shape[0] : 0,
          next_h_shape.size() > 1 ? next_h_shape[1] : 0,
          next_h_shape.size() > 2 ? next_h_shape[2] : 0);
      SHERPA_ONNX_EXIT(-1);
    }

    // next_c shape: (num_layers, 1, hidden_dim)
    auto next_c_shape = decoder_->TensorShape(decoder_output_c_name_);
    if (next_c_shape.size() != 3 || next_c_shape[0] != num_layers ||
        next_c_shape[1] != 1 || next_c_shape[2] != hidden_dim) {
      SHERPA_ONNX_LOGE(
          "Expected next_c shape (%d, 1, %d). Actual: [%d, %d, %d]",
          num_layers, hidden_dim,
          next_c_shape.size() > 0 ? next_c_shape[0] : 0,
          next_c_shape.size() > 1 ? next_c_shape[1] : 0,
          next_c_shape.size() > 2 ? next_c_shape[2] : 0);
      SHERPA_ONNX_EXIT(-1);
    }

    auto out_shape = decoder_->TensorShape(decoder_output_name_);
    if (out_shape.size() != 3 || out_shape[0] != 1 || out_shape[1] != 1) {
      SHERPA_ONNX_LOGE(
          "The decoder output should be of shape [1, 1, decoder_dim]");
      SHERPA_ONNX_EXIT(-1);
    }
    decoder_out_dim_ = out_shape[2];

    if (decoder_out_dim_ <= 0 || h_size_ <= 0 || c_size_ <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid decoder shapes. decoder_out_dim: %d, h_size: %d, c_size: %d",
          decoder_out_dim_, h_size_, c_size_);
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("decoder input '%s': [%d, %d]",
                       decoder_input_y_name_.c_str(), 1, 1);
      SHERPA_ONNX_LOGE("decoder input '%s': [%d, %d, %d]",
                       decoder_input_h_name_.c_str(), num_layers, hidden_dim,
                       1);
      SHERPA_ONNX_LOGE("decoder input '%s': [%d, %d, %d]",
                       decoder_input_c_name_.c_str(), num_layers, hidden_dim,
                       1);
      SHERPA_ONNX_LOGE("decoder output '%s': [%d, %d, %d]",
                       decoder_output_name_.c_str(), 1, 1, decoder_out_dim_);
      SHERPA_ONNX_LOGE("decoder output '%s': [%d, %d, %d]",
                       decoder_output_h_name_.c_str(), num_layers, 1,
                       hidden_dim);
      SHERPA_ONNX_LOGE("decoder output '%s': [%d, %d, %d]",
                       decoder_output_c_name_.c_str(), num_layers, 1,
                       hidden_dim);
    }
  }

  void CheckJoiner() {
    joiner_encoder_input_name_ = "encoder_out";
    joiner_decoder_input_name_ = "decoder_out";
    joiner_output_name_ = "log_probs";

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
    if (encoder_in_shape.size() != 3 || encoder_in_shape[0] != 1 ||
        encoder_in_shape[1] != encoder_out_dim_ || encoder_in_shape[2] != 1) {
      SHERPA_ONNX_LOGE(
          "The joiner input '%s' should be of shape [1, %d, 1]",
          joiner_encoder_input_name_.c_str(), encoder_out_dim_);
      SHERPA_ONNX_EXIT(-1);
    }

    auto decoder_in_shape = joiner_->TensorShape(joiner_decoder_input_name_);
    if (decoder_in_shape.size() != 3 || decoder_in_shape[0] != 1 ||
        decoder_in_shape[1] != decoder_out_dim_ || decoder_in_shape[2] != 1) {
      SHERPA_ONNX_LOGE(
          "The joiner input '%s' should be of shape [1, %d, 1]",
          joiner_decoder_input_name_.c_str(), decoder_out_dim_);
      SHERPA_ONNX_EXIT(-1);
    }

    auto out_shape = joiner_->TensorShape(joiner_output_name_);
    if (out_shape.size() != 4 || out_shape[0] != 1 || out_shape[1] != 1 ||
        out_shape[2] != 1) {
      SHERPA_ONNX_LOGE(
          "The joiner output should be of shape [1, 1, 1, vocab_size + "
          "num_durations]");
      SHERPA_ONNX_EXIT(-1);
    }

    // For TDT, the joiner output contains both token logits and duration logits
    // vocab_size + num_durations
    int32_t total_output = out_shape[3];
    if (total_output <= 0) {
      SHERPA_ONNX_LOGE("Invalid joiner output shape. total_output: %d",
                       total_output);
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("joiner input '%s': [%d, %d, %d]",
                       joiner_encoder_input_name_.c_str(), 1, encoder_out_dim_,
                       1);
      SHERPA_ONNX_LOGE("joiner input '%s': [%d, %d, %d]",
                       joiner_decoder_input_name_.c_str(), 1, decoder_out_dim_,
                       1);
      SHERPA_ONNX_LOGE("joiner output '%s': [%d, %d, %d, %d]",
                       joiner_output_name_.c_str(), 1, 1, 1, total_output);
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
  std::string decoder_input_y_name_;
  std::string decoder_input_h_name_;
  std::string decoder_input_c_name_;
  std::string decoder_output_name_;
  std::string decoder_output_h_name_;
  std::string decoder_output_c_name_;
  std::string joiner_encoder_input_name_;
  std::string joiner_decoder_input_name_;
  std::string joiner_output_name_;

  int32_t max_num_frames_ = 0;
  int32_t feat_dim_ = 0;
  int32_t max_num_encoder_frames_ = 0;
  int32_t encoder_out_dim_ = 0;
  int32_t decoder_out_dim_ = 0;
  int32_t h_size_ = 0;
  int32_t c_size_ = 0;
  int32_t subsampling_factor_ = 0;
};

OfflineParakeetTdtModelQnn::OfflineParakeetTdtModelQnn(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineParakeetTdtModelQnn::OfflineParakeetTdtModelQnn(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineParakeetTdtModelQnn::~OfflineParakeetTdtModelQnn() = default;

std::vector<float> OfflineParakeetTdtModelQnn::RunEncoder(
    std::vector<float> features) const {
  return impl_->RunEncoder(std::move(features));
}

std::vector<std::vector<float>>
OfflineParakeetTdtModelQnn::GetDecoderInitStates() const {
  return impl_->GetDecoderInitStates();
}

std::pair<std::vector<float>, std::vector<std::vector<float>>>
OfflineParakeetTdtModelQnn::RunDecoder(
    int32_t token, std::vector<std::vector<float>> states) const {
  return impl_->RunDecoder(token, std::move(states));
}

std::vector<float> OfflineParakeetTdtModelQnn::RunJoiner(
    const float *encoder_out, const std::vector<float> &decoder_out) const {
  return impl_->RunJoiner(encoder_out, decoder_out);
}

int32_t OfflineParakeetTdtModelQnn::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

int32_t OfflineParakeetTdtModelQnn::EncoderDim() const {
  return impl_->EncoderDim();
}

int32_t OfflineParakeetTdtModelQnn::FeatDim() const {
  return impl_->FeatDim();
}

int32_t OfflineParakeetTdtModelQnn::NumEncoderFrames(
    int32_t num_frames) const {
  return impl_->NumEncoderFrames(num_frames);
}

#if __ANDROID_API__ >= 9
template OfflineParakeetTdtModelQnn::OfflineParakeetTdtModelQnn(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineParakeetTdtModelQnn::OfflineParakeetTdtModelQnn(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
