// sherpa-onnx/csrc/qnn/offline-moonshine-model-qnn.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/offline-moonshine-model-qnn.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
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
#include "sherpa-onnx/csrc/qnn/qnn-backend.h"
#include "sherpa-onnx/csrc/qnn/qnn-model.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineMoonshineModelQnn::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    InitEncoder();
    InitDecoder();
    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    SHERPA_ONNX_LOGE(
        "Please copy all files from assets to SD card and set assetManager to "
        "null");
    SHERPA_ONNX_EXIT(-1);
  }

  OfflineMoonshineDecoderResult Run(
      const std::vector<float> &audio_samples) const {
    std::lock_guard<std::mutex> lock(mutex_);

    OfflineMoonshineDecoderResult r;

    if (audio_samples.empty()) {
      return r;
    }

    // Pad or truncate audio to fixed length
    std::vector<float> audio = audio_samples;
    if (static_cast<int32_t>(audio.size()) < max_audio_len_) {
      audio.resize(max_audio_len_, 0);
    } else if (static_cast<int32_t>(audio.size()) > max_audio_len_) {
      SHERPA_ONNX_LOGE(
          "Input audio length %d is too long. Truncate it to %d samples.",
          static_cast<int32_t>(audio.size()), max_audio_len_);

      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios or use VAD to cut your audio into "
          "small chunks.");

      audio.resize(max_audio_len_);
    }

    // Run encoder
    RunEncoder(audio);

    // Get cross KV from encoder and transpose for QNN
    // QNN expects (1, hidden_size, enc_seq_len) instead of (1, enc_seq_len, hidden_size)
    std::vector<std::vector<float>> cross_kv(num_layers_ * 2);
    for (int32_t i = 0; i < num_layers_; ++i) {
      auto cross_k = encoder_model_->GetOutputTensorData(encoder_cross_k_[i]);
      auto cross_v = encoder_model_->GetOutputTensorData(encoder_cross_v_[i]);

      cross_kv[i * 2] = Transpose(cross_k.data(), enc_seq_len_, hidden_size_);
      cross_kv[i * 2 + 1] = Transpose(cross_v.data(), enc_seq_len_, hidden_size_);
    }

    // Initialize self KV cache (pre-allocated, all zeros)
    // QNN cache shape: (1, hidden_size, max_seq_len)
    std::vector<float> self_kv(num_layers_ * 2 * self_kv_size_, 0);

    // Initialize mask (all ones = all masked)
    std::vector<int32_t> mask(mask_size_, 1);

    // Decode loop
    int32_t offset = 0;
    int32_t token_id = bos_;
    std::vector<float> logits;

    // Assume ~15 tokens per second, with safety limit
    int32_t active_audio_len = std::min(static_cast<int32_t>(audio.size()), max_audio_len_);
    int32_t max_tokens = std::min(mask_size_,
                                   static_cast<int32_t>(active_audio_len / 16000.0 * 15));
    max_tokens = std::min(max_tokens, mask_size_ - 1);

    while (offset < max_tokens) {
      logits = RunDecoder(token_id, offset, cross_kv, self_kv.data(), mask);

      token_id = MaxElementIndex(logits.data(), logits.size());

      if (token_id == eos_) {
        break;
      }

      r.tokens.push_back(token_id);

      UpdateSelfKvCache(self_kv.data(), offset);
      mask[offset] = 0;
      offset += 1;
    }

    return r;
  }

  int32_t MaxSeqLen() const { return mask_size_; }

 private:
  void InitEncoder() {
    const auto &qnn_config = config_.moonshine.qnn_config;

    encoder_backend_ = std::make_unique<QnnBackend>(qnn_config.backend_lib,
                                                    config_.debug);

    const auto &context_binary = qnn_config.context_binary;

    std::vector<std::string> binary_filenames;
    SplitStringToVector(context_binary, ",", true, &binary_filenames);

    std::string encoder_binary;
    if (binary_filenames.size() >= 1) {
      encoder_binary = binary_filenames[0];
    }

    if (!encoder_binary.empty() && FileExists(encoder_binary)) {
      // Use context binary directly
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Init encoder from context binary '%s'",
                         encoder_binary.c_str());
      }

      const auto &system_lib = qnn_config.system_lib;
      encoder_model_ = std::make_unique<QnnModel>(encoder_binary, system_lib,
                                                   encoder_backend_.get(),
                                                   BinaryContextTag{},
                                                   config_.debug);
    } else {
      // Need model lib file
      const auto &encoder_path = config_.moonshine.encoder;
      if (encoder_path.empty()) {
        SHERPA_ONNX_LOGE(
            "Please provide --moonshine-encoder or context binary for encoder");
        SHERPA_ONNX_EXIT(-1);
      }

      encoder_backend_->InitContext();
      encoder_model_ = std::make_unique<QnnModel>(encoder_path,
                                                   encoder_backend_.get(),
                                                   config_.debug);

      // Save context binary if path specified but file doesn't exist
      if (!encoder_binary.empty() && !FileExists(encoder_binary)) {
        if (config_.debug) {
          SHERPA_ONNX_LOGE("Saving context binary to '%s'", encoder_binary.c_str());
        }
        encoder_model_->SaveBinaryContext(encoder_binary);
      }
    }
  }

  void InitDecoder() {
    const auto &qnn_config = config_.moonshine.qnn_config;

    decoder_backend_ = std::make_unique<QnnBackend>(qnn_config.backend_lib,
                                                    config_.debug);

    const auto &context_binary = qnn_config.context_binary;

    std::vector<std::string> binary_filenames;
    SplitStringToVector(context_binary, ",", true, &binary_filenames);

    std::string decoder_binary;
    if (binary_filenames.size() >= 2) {
      decoder_binary = binary_filenames[1];
    }

    if (!decoder_binary.empty() && FileExists(decoder_binary)) {
      // Use context binary directly
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Init decoder from context binary '%s'",
                         decoder_binary.c_str());
      }

      const auto &system_lib = qnn_config.system_lib;
      decoder_model_ = std::make_unique<QnnModel>(decoder_binary, system_lib,
                                                   decoder_backend_.get(),
                                                   BinaryContextTag{},
                                                   config_.debug);
    } else {
      // Need model lib file
      const auto &decoder_path = config_.moonshine.decoder;
      if (decoder_path.empty()) {
        SHERPA_ONNX_LOGE(
            "Please provide --moonshine-decoder or context binary for decoder");
        SHERPA_ONNX_EXIT(-1);
      }

      decoder_backend_->InitContext();
      decoder_model_ = std::make_unique<QnnModel>(decoder_path,
                                                   decoder_backend_.get(),
                                                   config_.debug);

      // Save context binary if path specified but file doesn't exist
      if (!decoder_binary.empty() && !FileExists(decoder_binary)) {
        if (config_.debug) {
          SHERPA_ONNX_LOGE("Saving context binary to '%s'", decoder_binary.c_str());
        }
        decoder_model_->SaveBinaryContext(decoder_binary);
      }
    }
  }

  void PostInit() {
    PostInitEncoder();
    PostInitDecoder();
    PreComputeTensorNames();
  }

  void PostInitEncoder() {
    // Get encoder input shape: audio (1, max_audio_len)
    auto audio_shape = encoder_model_->TensorShape("audio");
    if (audio_shape.size() < 2) {
      SHERPA_ONNX_LOGE("Invalid encoder audio tensor rank");
      SHERPA_ONNX_EXIT(-1);
    }
    max_audio_len_ = audio_shape[1];

    // Get encoder output shape: cross_k_0 (1, enc_seq_len, hidden_size)
    auto cross_k_shape = encoder_model_->TensorShape("cross_k_0");
    if (cross_k_shape.size() < 3) {
      SHERPA_ONNX_LOGE("Invalid encoder cross_k_0 tensor rank");
      SHERPA_ONNX_EXIT(-1);
    }
    enc_seq_len_ = cross_k_shape[1];
    hidden_size_ = cross_k_shape[2];

    // Count layers from encoder outputs
    const auto &output_names = encoder_model_->OutputTensorNames();
    if (output_names.empty() || output_names.size() % 2 != 0) {
      SHERPA_ONNX_LOGE("Invalid encoder cross-KV output count");
      SHERPA_ONNX_EXIT(-1);
    }
    num_layers_ = output_names.size() / 2;

    if (config_.debug) {
      SHERPA_ONNX_LOGE("max_audio_len_: %d", max_audio_len_);
      SHERPA_ONNX_LOGE("enc_seq_len_: %d", enc_seq_len_);
      SHERPA_ONNX_LOGE("hidden_size_: %d", hidden_size_);
      SHERPA_ONNX_LOGE("num_layers_: %d", num_layers_);
    }
  }

  void PostInitDecoder() {
    // Get decoder input/output info
    // Inputs: tokens, self_k_0..N, self_v_0..N, cross_k_0..N, cross_v_0..N, offset, mask
    // Outputs: logits, this_self_k_0..N, this_self_v_0..N

    int32_t expected_inputs = 1 + num_layers_ * 2 + num_layers_ * 2 + 2;
    int32_t actual_inputs = decoder_model_->InputTensorNames().size();
    if (actual_inputs != expected_inputs) {
      SHERPA_ONNX_LOGE("Expected %d decoder inputs, got %d",
                       expected_inputs, actual_inputs);
      SHERPA_ONNX_EXIT(-1);
    }

    // Get mask size
    auto mask_shape = decoder_model_->TensorShape("mask");
    if (mask_shape.size() < 1) {
      SHERPA_ONNX_LOGE("Invalid decoder mask tensor rank");
      SHERPA_ONNX_EXIT(-1);
    }
    mask_size_ = mask_shape[0];

    // Get vocab size from logits
    auto logits_shape = decoder_model_->TensorShape("logits");
    if (logits_shape.size() < 3) {
      SHERPA_ONNX_LOGE("Invalid decoder logits tensor rank");
      SHERPA_ONNX_EXIT(-1);
    }
    vocab_size_ = logits_shape[2];

    // Self KV cache size: max_seq_len * hidden_size
    // QNN cache shape: (1, hidden_size, max_seq_len)
    self_kv_size_ = mask_size_ * hidden_size_;
    self_kv_stride_ = mask_size_;  // stride for the seq dimension

    if (config_.debug) {
      SHERPA_ONNX_LOGE("mask_size_: %d", mask_size_);
      SHERPA_ONNX_LOGE("vocab_size_: %d", vocab_size_);
      SHERPA_ONNX_LOGE("self_kv_size_: %d", self_kv_size_);
      SHERPA_ONNX_LOGE("self_kv_stride_: %d", self_kv_stride_);
    }
  }

  void PreComputeTensorNames() {
    encoder_cross_k_.resize(num_layers_);
    encoder_cross_v_.resize(num_layers_);
    decoder_cross_k_.resize(num_layers_);
    decoder_cross_v_.resize(num_layers_);
    decoder_self_k_.resize(num_layers_);
    decoder_self_v_.resize(num_layers_);
    decoder_this_self_k_.resize(num_layers_);
    decoder_this_self_v_.resize(num_layers_);

    for (int32_t i = 0; i < num_layers_; ++i) {
      std::string index = std::to_string(i);

      encoder_cross_k_[i] = "cross_k_" + index;
      encoder_cross_v_[i] = "cross_v_" + index;

      decoder_cross_k_[i] = "cross_k_" + index;
      decoder_cross_v_[i] = "cross_v_" + index;
      decoder_self_k_[i] = "self_k_" + index;
      decoder_self_v_[i] = "self_v_" + index;

      decoder_this_self_k_[i] = "this_self_k_" + index;
      decoder_this_self_v_[i] = "this_self_v_" + index;
    }
  }

  void RunEncoder(const std::vector<float> &audio) const {
    encoder_model_->SetInputTensorData("audio", audio.data(), audio.size());
    encoder_model_->Run();
  }

  std::vector<float> RunDecoder(
      int32_t token, int32_t offset,
      const std::vector<std::vector<float>> &cross_kv,
      const float *self_kv_data,
      const std::vector<int32_t> &mask) const {
    // Set tokens
    decoder_model_->SetInputTensorData("tokens", &token, 1);

    // Set self kv cache
    for (int32_t i = 0; i < num_layers_; ++i) {
      decoder_model_->SetInputTensorData(decoder_self_k_[i],
                                         self_kv_data + (i * 2) * self_kv_size_,
                                         self_kv_size_);

      decoder_model_->SetInputTensorData(decoder_self_v_[i],
                                         self_kv_data + (i * 2 + 1) * self_kv_size_,
                                         self_kv_size_);
    }

    // Set cross kv cache (transposed)
    for (int32_t i = 0; i < num_layers_; ++i) {
      decoder_model_->SetInputTensorData(decoder_cross_k_[i],
                                         cross_kv[i * 2].data(),
                                         cross_kv[i * 2].size());

      decoder_model_->SetInputTensorData(decoder_cross_v_[i],
                                         cross_kv[i * 2 + 1].data(),
                                         cross_kv[i * 2 + 1].size());
    }

    // Set offset
    decoder_model_->SetInputTensorData("offset", &offset, 1);

    // Set mask
    decoder_model_->SetInputTensorData("mask", mask.data(), mask.size());

    // Run decoder
    decoder_model_->Run();

    // Get logits
    return decoder_model_->GetOutputTensorData("logits");
  }

  void UpdateSelfKvCache(float *self_kv_data, int32_t offset) const {
    for (int32_t i = 0; i < num_layers_; ++i) {
      // Update self_k
      auto delta_k = decoder_model_->GetOutputTensorData(decoder_this_self_k_[i]);
      float *self_k = self_kv_data + (i * 2) * self_kv_size_;
      for (size_t r = 0; r != delta_k.size(); ++r) {
        self_k[r * self_kv_stride_ + offset] += delta_k[r];
      }

      // Update self_v
      auto delta_v = decoder_model_->GetOutputTensorData(decoder_this_self_v_[i]);
      float *self_v = self_kv_data + (i * 2 + 1) * self_kv_size_;
      for (size_t r = 0; r != delta_v.size(); ++r) {
        self_v[r * self_kv_stride_ + offset] += delta_v[r];
      }
    }
  }

 private:
  mutable std::mutex mutex_;
  OfflineModelConfig config_;

  std::unique_ptr<QnnBackend> encoder_backend_;
  std::unique_ptr<QnnModel> encoder_model_;

  std::unique_ptr<QnnBackend> decoder_backend_;
  std::unique_ptr<QnnModel> decoder_model_;

  // Dimensions
  int32_t max_audio_len_ = 0;
  int32_t enc_seq_len_ = 0;
  int32_t hidden_size_ = 0;
  int32_t num_layers_ = 0;
  int32_t mask_size_ = 0;
  int32_t vocab_size_ = 0;
  int32_t self_kv_size_ = 0;
  int32_t self_kv_stride_ = 0;

  // Pre-computed tensor names
  std::vector<std::string> encoder_cross_k_;
  std::vector<std::string> encoder_cross_v_;
  std::vector<std::string> decoder_cross_k_;
  std::vector<std::string> decoder_cross_v_;
  std::vector<std::string> decoder_self_k_;
  std::vector<std::string> decoder_self_v_;
  std::vector<std::string> decoder_this_self_k_;
  std::vector<std::string> decoder_this_self_v_;

  // Special tokens
  int32_t bos_ = 1;
  int32_t eos_ = 2;
};

OfflineMoonshineModelQnn::OfflineMoonshineModelQnn(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineMoonshineModelQnn::OfflineMoonshineModelQnn(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineMoonshineModelQnn::~OfflineMoonshineModelQnn() = default;

OfflineMoonshineDecoderResult OfflineMoonshineModelQnn::Run(
    const std::vector<float> &audio_samples) const {
  return impl_->Run(audio_samples);
}

int32_t OfflineMoonshineModelQnn::MaxSeqLen() const {
  return impl_->MaxSeqLen();
}

#if __ANDROID_API__ >= 9
template OfflineMoonshineModelQnn::OfflineMoonshineModelQnn(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineMoonshineModelQnn::OfflineMoonshineModelQnn(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
