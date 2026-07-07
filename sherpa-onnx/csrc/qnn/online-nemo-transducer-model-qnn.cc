// sherpa-onnx/csrc/qnn/online-nemo-transducer-model-qnn.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// QNN layout differences from ONNX:
//
//   Encoder input x:  QNN uses [1, window_size, feat_dim]
//                     (ONNX uses [1, feat_dim, window_size])
//
//   Encoder cache:    QNN input [1, d1, d2, d0], QNN output [1, d0, d1, d2]
//                     (ONNX uses [1, d0, d1, d2])
//                     Storage uses QNN input format. Transpose012To120 is
//                     applied once on output; no transpose on input.

#include "sherpa-onnx/csrc/qnn/online-nemo-transducer-model-qnn.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <numeric>
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

namespace {

// Transpose a 3D tensor from [d0, d1, d2] to [d1, d2, d0].
// Used to convert QNN encoder cache outputs back to the storage layout
// expected by the next encoder call.
// Loop order: i1 (outer), i2 (middle), i0 (inner).
// This makes writes to y sequential (consecutive i0 values are contiguous).
static std::vector<float> Transpose012To120(const float *x, int64_t d0,
                                            int64_t d1, int64_t d2) {
  std::vector<float> y(static_cast<size_t>(d0) * d1 * d2);

  for (int64_t i1 = 0; i1 != d1; ++i1) {
    for (int64_t i2 = 0; i2 != d2; ++i2) {
      const float *src = x + (i1 * d2 + i2);
      float *dst = y.data() + (i1 * d2 + i2) * d0;
      for (int64_t i0 = 0; i0 != d0; ++i0) {
        dst[i0] = src[i0 * d1 * d2];
      }
    }
  }

  return y;
}

static int64_t NumElements(const std::vector<int64_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

}  // namespace

class OnlineNemoTransducerModelQnn::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config) : config_(config) { Init(); }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config) : config_(config) {
    SHERPA_ONNX_LOGE(
        "Please copy all files from assets to SD card and set assetManager to "
        "null");
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<OnlineStreamStateTensor> GetEncoderInitStates() const {
    std::vector<OnlineStreamStateTensor> ans;
    ans.reserve(state_storage_shapes_.size());
    for (size_t i = 0; i != state_storage_shapes_.size(); ++i) {
      OnlineStreamStateTensor state;
      int64_t n = NumElements(state_storage_shapes_[i]);
      if (state_is_int32_[i]) {
        state.int32_data.resize(n, 0);
      } else {
        state.float_data.resize(n, 0.f);
      }
      ans.push_back(std::move(state));
    }

    return ans;
  }

  // QNN encoder tensor shapes:
  //
  //   Inputs:
  //     x_{window_size}_{window_shift}[_norm]: (1, window_size, feat_dim) float
  //     cache_last_channel:  (1, 70, 1024, 24) float  (QNN input layout)
  //     cache_last_time:     (1, 1024, 8, 24) float    (QNN input layout)
  //     cache_last_channel_len: (1,) int32
  //
  //   Outputs:
  //     encoder_out:              (1, num_encoder_frames, encoder_out_dim) float
  //     next_cache_last_channel:  (1, 24, 70, 1024) float
  //     next_cache_last_time:     (1, 24, 1024, 8) float
  //     next_cache_last_channel_len: (1,) int32
  //
  //   Cache layout: QNN output uses (d0, d1, d2), QNN input uses (d1, d2, d0).
  //   We transpose once on output (Transpose012To120) and store in QNN input
  //   format, so no transpose is needed on the next call.
  std::vector<float> RunEncoder(std::vector<float> features, int32_t num_frames,
                                std::vector<OnlineStreamStateTensor> *states) const {
    if (!states) {
      SHERPA_ONNX_LOGE("states pointer is null");
      SHERPA_ONNX_EXIT(-1);
    }

    if (states->size() < state_input_names_.size()) {
      SHERPA_ONNX_LOGE("states size %d is less than expected %d",
                        static_cast<int32_t>(states->size()),
                        static_cast<int32_t>(state_input_names_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Features come in as [num_frames, feat_dim] row-major, which matches
    // the QNN encoder input layout [1, window_size, feat_dim] directly.
    encoder_->SetInputTensorData(encoder_input_name_, features.data(),
                                 static_cast<int32_t>(features.size()));

    // Set encoder cache state inputs.
    // Data is stored in QNN input format (transposed once on output),
    // so feed directly without transpose.
    for (size_t i = 0; i != state_input_names_.size(); ++i) {
      const auto &name = state_input_names_[i];
      if (!(*states)[i].int32_data.empty()) {
        const auto *p = (*states)[i].int32_data.data();
        int32_t n = static_cast<int32_t>((*states)[i].int32_data.size());
        encoder_->SetInputTensorData(name, p, n);
      } else {
        const auto *p = (*states)[i].float_data.data();
        int32_t n = static_cast<int32_t>((*states)[i].float_data.size());
        encoder_->SetInputTensorData(name, p, n);
      }
    }

    encoder_->Run();

    std::vector<float> encoder_out =
        encoder_->GetOutputTensorData("encoder_out");

    // Read encoder cache outputs. Transpose once to QNN input format
    // so it can be fed directly on the next call without transpose.
    for (size_t i = 0; i != state_output_names_.size(); ++i) {
      const auto &name = state_output_names_[i];
      if (state_is_int32_[i]) {
        (*states)[i].int32_data = encoder_->GetOutputTensorDataInt32(name);
      } else {
        auto data = encoder_->GetOutputTensorData(name);
        if (state_needs_transpose_[i]) {
          const auto &out_shape = state_output_shapes_[i];
          data = Transpose012To120(data.data(), out_shape[1], out_shape[2],
                                      out_shape[3]);
        }
        (*states)[i].float_data = std::move(data);
      }
    }

    return encoder_out;
  }

  // QNN decoder tensor shapes:
  //
  //   Inputs:
  //     y: (1, 1) int32
  //     h: (num_layers, hidden_dim, 1) float
  //     c: (num_layers, hidden_dim, 1) float
  //
  //   Outputs:
  //     decoder_out: (1, 1, decoder_out_dim) float
  //     next_h:      (num_layers, 1, hidden_dim) float
  //     next_c:      (num_layers, 1, hidden_dim) float
  //
  //   Note: h/c input layout (d0, d1, 1) differs from next_h/next_c output
  //   layout (d0, 1, d1), but since one dim is 1 the flat buffer is the same.
  std::pair<std::vector<float>, std::vector<std::vector<float>>>
  RunDecoder(int32_t token, std::vector<std::vector<float>> states) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (states.size() < 2) {
      SHERPA_ONNX_LOGE("states should have at least 2 elements (h and c)");
      SHERPA_ONNX_EXIT(-1);
    }

    // states[0] = h, states[1] = c
    decoder_->SetInputTensorData("y", &token, 1);
    decoder_->SetInputTensorData("h", states[0].data(),
                                 static_cast<int32_t>(states[0].size()));
    decoder_->SetInputTensorData("c", states[1].data(),
                                 static_cast<int32_t>(states[1].size()));
    decoder_->Run();

    std::vector<float> decoder_out =
        decoder_->GetOutputTensorData("decoder_out");
    states[0] = decoder_->GetOutputTensorData("next_h");
    states[1] = decoder_->GetOutputTensorData("next_c");

    return {std::move(decoder_out), std::move(states)};
  }

  // QNN joiner tensor shapes:
  //
  //   Inputs:
  //     encoder_out: (1, encoder_out_dim, 1) float
  //     decoder_out: (1, decoder_out_dim, 1) float
  //
  //   Outputs:
  //     logits: (1, 1, 1, vocab_size) float
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

  int32_t WindowSize() const { return window_size_; }
  int32_t WindowShift() const { return window_shift_; }
  int32_t VocabSize() const { return vocab_size_; }
  int32_t FeatureDim() const { return feat_dim_; }
  int32_t EncoderDim() const { return encoder_out_dim_; }
  int32_t DecoderDim() const { return decoder_out_dim_; }
  int32_t SubsamplingFactor() const { return subsampling_factor_; }
  const std::string &NormalizationType() const { return normalization_type_; }

  std::vector<std::vector<float>> GetDecoderInitState() const {
    return {std::vector<float>(h_size_, 0.f),
            std::vector<float>(c_size_, 0.f)};
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
    SplitStringToVector(config_.transducer.qnn_config.context_binary, ",",
                        true, &context_binaries_);
    if (!context_binaries_.empty() && context_binaries_.size() != 3) {
      SHERPA_ONNX_LOGE(
          "There should be 3 files for online parakeet TDT context binary. "
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
      SHERPA_ONNX_LOGE("Saved context binary to '%s'.",
                        context_binary.c_str());
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
    InitComponent(config_.transducer.encoder,
                  context_binaries_.empty() ? "" : context_binaries_[0],
                  "encoder", encoder_backend_.get(), &encoder_);
  }

  void InitDecoder() {
    InitComponent(config_.transducer.decoder,
                  context_binaries_.empty() ? "" : context_binaries_[1],
                  "decoder", decoder_backend_.get(), &decoder_);
  }

  void InitJoiner() {
    InitComponent(config_.transducer.joiner,
                  context_binaries_.empty() ? "" : context_binaries_[2],
                  "joiner", joiner_backend_.get(), &joiner_);
  }

  void CheckEncoder() {
    if (!encoder_->HasTensor("encoder_out")) {
      SHERPA_ONNX_LOGE("The encoder model does not have output 'encoder_out'");
      SHERPA_ONNX_EXIT(-1);
    }

    // The encoder input has a dynamic name: x_{window_size}_{window_shift}
    // e.g., x_121_112
    encoder_input_name_.clear();
    for (const auto &name : encoder_->InputTensorNames()) {
      if (name.size() > 2 && name[0] == 'x' && name[1] == '_') {
        encoder_input_name_ = name;
        std::vector<std::string> parts;
        SplitStringToVector(name, "_", false, &parts);
        // x name can be:
        //   x_window-size_window-shift
        //   x_window-size_window-shift_normalization-method
        // where normalization-method is "NA" or "per-feature"
        if (parts.size() != 3 && parts.size() != 4) {
          SHERPA_ONNX_LOGE(
              "Expected encoder input name "
              "'x_window-size_window-shift[_normalization-method]'. "
              "Given: '%s'",
              name.c_str());
          SHERPA_ONNX_EXIT(-1);
        }

        window_size_ = std::atoi(parts[1].c_str());
        window_shift_ = std::atoi(parts[2].c_str());

        if (window_size_ <= 0 || window_shift_ <= 0) {
          SHERPA_ONNX_LOGE(
              "Invalid window size/shift in encoder input name '%s'",
              name.c_str());
          SHERPA_ONNX_EXIT(-1);
        }

        normalization_type_.clear();
        if (parts.size() == 4) {
          if (parts[3] == "per-feature") {
            normalization_type_ = "per_feature";
          } else if (parts[3] != "NA") {
            SHERPA_ONNX_LOGE(
                "Unknown normalization method '%s' in encoder input name '%s'. "
                "Expected 'NA' or 'per-feature'.",
                parts[3].c_str(), name.c_str());
            SHERPA_ONNX_EXIT(-1);
          }
        }

        break;
      }
    }

    if (encoder_input_name_.empty()) {
      SHERPA_ONNX_LOGE(
          "The encoder model does not have an input matching 'x_*_*'");
      SHERPA_ONNX_EXIT(-1);
    }

    // Identify cache state inputs (cache_last_channel, cache_last_time,
    // cache_last_channel_len).
    state_input_names_.clear();
    for (const auto &name : encoder_->InputTensorNames()) {
      if (name == encoder_input_name_) {
        continue;
      }
      if (name == "cache_last_channel" || name == "cache_last_time" ||
          name == "cache_last_channel_len") {
        state_input_names_.push_back(name);
      }
    }

    if (state_input_names_.size() != 3) {
      SHERPA_ONNX_LOGE(
          "Expected 3 encoder cache states (cache_last_channel, "
          "cache_last_time, cache_last_channel_len). Found %d",
          static_cast<int32_t>(state_input_names_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    // The output cache names use "next_" prefix.
    state_output_names_.clear();
    for (const auto &name : state_input_names_) {
      std::string out_name = "next_" + name;
      if (!encoder_->HasTensor(out_name)) {
        SHERPA_ONNX_LOGE("The encoder model does not have output '%s'",
                         out_name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      state_output_names_.push_back(out_name);
    }

    // Validate input x shape: [1, window_size, feat_dim] (QNN layout).
    // The ONNX model uses [1, feat_dim, window_size] but the QNN converter
    // changes the layout to [1, window_size, feat_dim].
    auto x_shape = encoder_->TensorShape(encoder_input_name_);
    if (x_shape.size() != 3 || x_shape[0] != 1 ||
        x_shape[1] != window_size_) {
      SHERPA_ONNX_LOGE(
          "The encoder input x should be of shape [1, %d, feat_dim]. "
          "Actual: [1, %d, %d]",
          window_size_, x_shape[1], x_shape[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    feat_dim_ = x_shape[2];

    auto out_shape = encoder_->TensorShape("encoder_out");
    if (out_shape.size() != 3 || out_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The encoder output should be of shape [1, ?, ?]");
      SHERPA_ONNX_EXIT(-1);
    }

    encoder_out_dim_ = out_shape[2];
    num_encoder_frames_ = out_shape[1];

    // Each chunk of window_shift input frames produces num_encoder_frames_
    // output frames, so the subsampling factor is window_shift /
    // num_encoder_frames_.
    subsampling_factor_ = window_shift_ / num_encoder_frames_;
    if (subsampling_factor_ <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid subsampling factor: %d (window_shift=%d, "
          "num_encoder_frames=%d)",
          subsampling_factor_, window_shift_, num_encoder_frames_);
      SHERPA_ONNX_EXIT(-1);
    }

    // Validate and store cache shapes.
    // cache_last_channel and cache_last_time are 4D and need 3D transpose.
    // cache_last_channel_len is 1D int32, no transpose.
    for (size_t i = 0; i != state_input_names_.size(); ++i) {
      const auto &name = state_input_names_[i];
      auto in_shape = encoder_->TensorShape(name);
      auto out_shape = encoder_->TensorShape(state_output_names_[i]);

      state_storage_shapes_.emplace_back(in_shape.begin(), in_shape.end());
      state_output_shapes_.emplace_back(out_shape.begin(), out_shape.end());

      bool is_int32 = (name == "cache_last_channel_len");
      state_is_int32_.push_back(is_int32);

      // cache_last_channel and cache_last_time need 3D transpose
      // (model output [d0,d1,d2] <-> storage [d1,d2,d0])
      bool needs_transpose =
          (name == "cache_last_channel" || name == "cache_last_time");
      state_needs_transpose_.push_back(needs_transpose);

      if (needs_transpose) {
        // Validate that the transpose is consistent:
        // input [1, d1, d2, d0] -> output [1, d0, d1, d2]
        if (in_shape.size() != 4 || out_shape.size() != 4 ||
            in_shape[0] != 1 || out_shape[0] != 1 ||
            in_shape[1] != out_shape[2] || in_shape[2] != out_shape[3] ||
            in_shape[3] != out_shape[1]) {
          SHERPA_ONNX_LOGE(
              "Inconsistent cache shapes for '%s': input [%d,%d,%d,%d] "
              "output [%d,%d,%d,%d]",
              name.c_str(), in_shape[0], in_shape[1], in_shape[2],
              in_shape[3], out_shape[0], out_shape[1], out_shape[2],
              out_shape[3]);
          SHERPA_ONNX_LOGE(
              "Expected: input [1, d1, d2, d0] -> output [1, d0, d1, d2]");
          SHERPA_ONNX_EXIT(-1);
        }
      }
    }
  }

  void CheckDecoder() {
    // Parakeet TDT decoder: 3 inputs (y, h, c), 3 outputs (decoder_out,
    // next_h, next_c)
    if (!decoder_->HasTensor("y")) {
      SHERPA_ONNX_LOGE("The decoder model does not have input 'y'");
      SHERPA_ONNX_EXIT(-1);
    }
    if (!decoder_->HasTensor("h")) {
      SHERPA_ONNX_LOGE("The decoder model does not have input 'h'");
      SHERPA_ONNX_EXIT(-1);
    }
    if (!decoder_->HasTensor("c")) {
      SHERPA_ONNX_LOGE("The decoder model does not have input 'c'");
      SHERPA_ONNX_EXIT(-1);
    }
    if (!decoder_->HasTensor("decoder_out")) {
      SHERPA_ONNX_LOGE("The decoder model does not have output 'decoder_out'");
      SHERPA_ONNX_EXIT(-1);
    }
    if (!decoder_->HasTensor("next_h")) {
      SHERPA_ONNX_LOGE("The decoder model does not have output 'next_h'");
      SHERPA_ONNX_EXIT(-1);
    }
    if (!decoder_->HasTensor("next_c")) {
      SHERPA_ONNX_LOGE("The decoder model does not have output 'next_c'");
      SHERPA_ONNX_EXIT(-1);
    }

    auto y_shape = decoder_->TensorShape("y");
    if (y_shape.size() != 2 || y_shape[0] != 1 || y_shape[1] != 1) {
      SHERPA_ONNX_LOGE(
          "Expected decoder input 'y' shape [1, 1]. Actual: [%d, %d]",
          y_shape[0], y_shape.size() > 1 ? y_shape[1] : 0);
      SHERPA_ONNX_EXIT(-1);
    }

    // QNN decoder h/c input:  (num_layers, hidden_dim, 1)
    // QNN decoder next_h/next_c output: (num_layers, 1, hidden_dim)
    auto h_shape = decoder_->TensorShape("h");
    if (h_shape.size() != 3 || h_shape[2] != 1) {
      SHERPA_ONNX_LOGE(
          "Expected decoder input 'h' shape [num_layers, hidden_dim, 1]. "
          "Actual: [%d, %d, %d]",
          h_shape.size() > 0 ? h_shape[0] : 0,
          h_shape.size() > 1 ? h_shape[1] : 0,
          h_shape.size() > 2 ? h_shape[2] : 0);
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t num_layers = h_shape[0];
    int32_t hidden_dim = h_shape[1];
    h_size_ = num_layers * hidden_dim;

    auto c_shape = decoder_->TensorShape("c");
    if (c_shape.size() != 3 || c_shape[0] != num_layers || c_shape[2] != 1) {
      SHERPA_ONNX_LOGE(
          "Expected decoder input 'c' shape [%d, ?, 1]. Actual: [%d, %d, %d]",
          num_layers,
          c_shape.size() > 0 ? c_shape[0] : 0,
          c_shape.size() > 1 ? c_shape[1] : 0,
          c_shape.size() > 2 ? c_shape[2] : 0);
      SHERPA_ONNX_EXIT(-1);
    }
    c_size_ = c_shape[0] * c_shape[1];

    auto out_shape = decoder_->TensorShape("decoder_out");
    if (out_shape.size() != 3 || out_shape[0] != 1 || out_shape[1] != 1) {
      SHERPA_ONNX_LOGE(
          "Expected decoder output 'decoder_out' shape [1, 1, decoder_dim]. "
          "Actual: [%d, %d, %d]",
          out_shape.size() > 0 ? out_shape[0] : 0,
          out_shape.size() > 1 ? out_shape[1] : 0,
          out_shape.size() > 2 ? out_shape[2] : 0);
      SHERPA_ONNX_EXIT(-1);
    }
    decoder_out_dim_ = out_shape[2];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("decoder input 'y': [1, 1]");
      SHERPA_ONNX_LOGE("decoder input 'h': [%d, 1, %d]", num_layers,
                        hidden_dim);
      SHERPA_ONNX_LOGE("decoder input 'c': [%d, 1, %d]", num_layers,
                        hidden_dim);
      SHERPA_ONNX_LOGE("decoder output 'decoder_out': [1, 1, %d]",
                        decoder_out_dim_);
    }
  }

  void CheckJoiner() {
    joiner_encoder_input_name_ = "encoder_out";
    joiner_decoder_input_name_ = "decoder_out";

    joiner_output_name_ = "logits";

    if (!joiner_->HasTensor(joiner_output_name_)) {
      SHERPA_ONNX_LOGE("The joiner model does not have output '%s'",
                       joiner_output_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

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

    // QNN joiner input shapes: encoder_out = (1, encoder_out_dim, 1),
    // decoder_out = (1, decoder_out_dim, 1)
    auto encoder_in_shape = joiner_->TensorShape(joiner_encoder_input_name_);
    if (encoder_in_shape.size() != 3 || encoder_in_shape[0] != 1 ||
        encoder_in_shape[1] != encoder_out_dim_ || encoder_in_shape[2] != 1) {
      SHERPA_ONNX_LOGE(
          "The joiner input '%s' should be [1, %d, 1]. Actual: [%d, %d, %d]",
          joiner_encoder_input_name_.c_str(), encoder_out_dim_,
          encoder_in_shape[0], encoder_in_shape[1], encoder_in_shape[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    auto decoder_in_shape = joiner_->TensorShape(joiner_decoder_input_name_);
    if (decoder_in_shape.size() != 3 || decoder_in_shape[0] != 1 ||
        decoder_in_shape[1] != decoder_out_dim_ || decoder_in_shape[2] != 1) {
      SHERPA_ONNX_LOGE(
          "The joiner input '%s' should be [1, %d, 1]. Actual: [%d, %d, %d]",
          joiner_decoder_input_name_.c_str(), decoder_out_dim_,
          decoder_in_shape[0], decoder_in_shape[1], decoder_in_shape[2]);
      SHERPA_ONNX_EXIT(-1);
    }

    auto out_shape = joiner_->TensorShape(joiner_output_name_);
    // The joiner output can be 2D [1, vocab_size] or 4D [1, 1, 1, vocab_size]
    // depending on the QNN converter. The last dimension is always vocab_size.
    if (out_shape.size() < 2 || out_shape[0] != 1) {
      SHERPA_ONNX_LOGE(
          "The joiner output should have first dimension 1. Actual shape "
          "rank: %d",
          static_cast<int32_t>(out_shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    vocab_size_ = out_shape.back();

    if (config_.debug) {
      SHERPA_ONNX_LOGE("joiner input '%s': [1, %d]",
                        joiner_encoder_input_name_.c_str(), encoder_out_dim_);
      SHERPA_ONNX_LOGE("joiner input '%s': [1, %d]",
                        joiner_decoder_input_name_.c_str(), decoder_out_dim_);
      SHERPA_ONNX_LOGE("joiner output '%s': rank %d, vocab_size %d",
                        joiner_output_name_.c_str(),
                        static_cast<int32_t>(out_shape.size()), vocab_size_);
    }
  }

 private:
  mutable std::mutex mutex_;

  OnlineModelConfig config_;

  std::unique_ptr<QnnBackend> encoder_backend_;
  std::unique_ptr<QnnBackend> decoder_backend_;
  std::unique_ptr<QnnBackend> joiner_backend_;

  std::unique_ptr<QnnModel> encoder_;
  std::unique_ptr<QnnModel> decoder_;
  std::unique_ptr<QnnModel> joiner_;
  std::vector<std::string> context_binaries_;

  std::string encoder_input_name_;
  std::vector<std::string> state_input_names_;
  std::vector<std::string> state_output_names_;

  std::string joiner_encoder_input_name_;
  std::string joiner_decoder_input_name_;
  std::string joiner_output_name_;

  std::vector<std::vector<int64_t>> state_storage_shapes_;
  std::vector<std::vector<int64_t>> state_output_shapes_;
  std::vector<bool> state_is_int32_;
  std::vector<bool> state_needs_transpose_;

  int32_t window_size_ = 0;
  int32_t window_shift_ = 0;
  int32_t num_encoder_frames_ = 0;
  int32_t subsampling_factor_ = 0;
  int32_t feat_dim_ = 0;
  int32_t encoder_out_dim_ = 0;
  int32_t decoder_out_dim_ = 0;
  int32_t vocab_size_ = 0;
  int32_t h_size_ = 0;
  int32_t c_size_ = 0;
  std::string normalization_type_;  // "" or "per_feature"
};

OnlineNemoTransducerModelQnn::OnlineNemoTransducerModelQnn(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineNemoTransducerModelQnn::OnlineNemoTransducerModelQnn(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OnlineNemoTransducerModelQnn::~OnlineNemoTransducerModelQnn() = default;

std::vector<OnlineStreamStateTensor>
OnlineNemoTransducerModelQnn::GetEncoderInitStates() const {
  return impl_->GetEncoderInitStates();
}

std::vector<float> OnlineNemoTransducerModelQnn::RunEncoder(
    std::vector<float> features, int32_t num_frames,
    std::vector<OnlineStreamStateTensor> *states) const {
  return impl_->RunEncoder(std::move(features), num_frames, states);
}

std::pair<std::vector<float>, std::vector<std::vector<float>>>
OnlineNemoTransducerModelQnn::RunDecoder(
    int32_t token, std::vector<std::vector<float>> states) const {
  return impl_->RunDecoder(token, std::move(states));
}

std::vector<float> OnlineNemoTransducerModelQnn::RunJoiner(
    const float *encoder_out, const std::vector<float> &decoder_out) const {
  return impl_->RunJoiner(encoder_out, decoder_out);
}

int32_t OnlineNemoTransducerModelQnn::WindowSize() const {
  return impl_->WindowSize();
}

int32_t OnlineNemoTransducerModelQnn::WindowShift() const {
  return impl_->WindowShift();
}

int32_t OnlineNemoTransducerModelQnn::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OnlineNemoTransducerModelQnn::FeatureDim() const {
  return impl_->FeatureDim();
}

int32_t OnlineNemoTransducerModelQnn::EncoderDim() const {
  return impl_->EncoderDim();
}

int32_t OnlineNemoTransducerModelQnn::DecoderDim() const {
  return impl_->DecoderDim();
}

int32_t OnlineNemoTransducerModelQnn::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

const std::string &OnlineNemoTransducerModelQnn::NormalizationType() const {
  return impl_->NormalizationType();
}

std::vector<std::vector<float>>
OnlineNemoTransducerModelQnn::GetDecoderInitState() const {
  return impl_->GetDecoderInitState();
}

#if __ANDROID_API__ >= 9
template OnlineNemoTransducerModelQnn::OnlineNemoTransducerModelQnn(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineNemoTransducerModelQnn::OnlineNemoTransducerModelQnn(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
