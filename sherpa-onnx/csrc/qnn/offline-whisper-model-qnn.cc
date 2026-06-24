// sherpa-onnx/csrc/qnn/offline-whisper-model-qnn.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/offline-whisper-model-qnn.h"

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
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-whisper-model-config.h"
#include "sherpa-onnx/csrc/qnn/macros.h"
#include "sherpa-onnx/csrc/qnn/qnn-backend.h"
#include "sherpa-onnx/csrc/qnn/qnn-model.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

// Parse the model name prefix from the first input tensor name.
// e.g., "tiny_en_mel" -> "tiny_en"
static std::string ParseModelPrefix(const std::string &input_name) {
  auto pos = input_name.rfind("_mel");
  if (pos == std::string::npos) {
    SHERPA_ONNX_LOGE("Unexpected encoder input name: '%s'", input_name.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  return input_name.substr(0, pos);
}

// Convert prefix like "tiny_en" to WhisperModelType
static WhisperModelType ParseWhisperModelFromPrefix(const std::string &prefix) {
  std::string name = prefix;
  auto pos = name.find('_');
  if (pos != std::string::npos) {
    name[pos] = '.';
  }
  return ParseWhisperModelType(name);
}

class OfflineWhisperModelQnn::Impl {
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

  OfflineWhisperDecoderResult Run(std::vector<float> features) const {
    std::lock_guard<std::mutex> lock(mutex_);

    OfflineWhisperDecoderResult r;

    if (features.empty()) {
      return r;
    }

    int32_t num_frames = features.size() / feat_dim_;
    if (num_frames > num_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          num_frames, num_frames_);

      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios or use VAD to cut your audio into "
          "small chunks.");

      num_frames = num_frames_;
    }

    // assume at most 6 tokens per second
    int32_t num_possible_tokens = num_frames / 100.0 * 6;
    num_possible_tokens =
        std::min<int32_t>(num_possible_tokens, mask_size_ / 2);

    // Pad to expected frames
    features.resize(num_frames_ * feat_dim_, 0);

    // For QNN, encoder input shape is (1, T, C), no transpose needed
    RunEncoder(features);

    // Get encoder outputs and transpose for decoder
    // cross_kv layout: [cross_k_0, cross_v_0, cross_k_1, cross_v_1, ...]
    std::vector<std::vector<float>> cross_kv(n_text_layer_ * 2);
    for (int32_t i = 0; i < n_text_layer_; ++i) {
      auto cross_k = encoder_model_->GetOutputTensorData(encoder_cross_k_[i]);
      auto cross_v = encoder_model_->GetOutputTensorData(encoder_cross_v_[i]);

      cross_kv[i * 2] =
          Transpose(cross_k.data(), num_out_frames_, n_text_state_);
      cross_kv[i * 2 + 1] =
          Transpose(cross_v.data(), num_out_frames_, n_text_state_);
    }

    // Initialize self kv cache to zeros (flat vector for better cache locality)
    // Layout: [layer0_k, layer0_v, layer1_k, layer1_v, ...]
    // Each layer has mask_size * n_text_state elements
    std::vector<float> self_kv(n_text_layer_ * 2 * self_kv_size_, 0);

    // Initialize mask (all ones)
    std::vector<int32_t> mask(mask_size_, 1);

    // Prepare sot_sequence
    std::vector<int32_t> sot_sequence(sot_sequence_);

    if (IsMultilingual(model_type_)) {
      if (config_.whisper.task == "translate") {
        sot_sequence[2] = translate_;
      } else if (config_.whisper.task != "transcribe") {
        SHERPA_ONNX_LOGE(
            "Valid task values are: translate, transcribe. Given: '%s'",
            config_.whisper.task.c_str());
        SHERPA_ONNX_EXIT(-1);
      }

      if (!config_.whisper.language.empty()) {
        int32_t lang_id = GetWhisperLanguageTokenId(config_.whisper.language);
        if (lang_id < 0) {
          SHERPA_ONNX_LOGE("Unsupported language: '%s'",
                           config_.whisper.language.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
        r.lang = config_.whisper.language;
        sot_sequence[1] = lang_id;
      } else {
        if (config_.debug) {
          SHERPA_ONNX_LOGE("Detecting language.");
        }

        int32_t lang_id = DetectLanguage(cross_kv, self_kv, mask);
        r.lang = GetWhisperLanguageCode(lang_id);

        if (config_.debug) {
          SHERPA_ONNX_LOGE("Detected Language: %s", r.lang.c_str());
        }

        sot_sequence[1] = lang_id;
      }
    }

    int32_t offset = 0;
    std::vector<float> logits;
    float *self_kv_data = self_kv.data();

    // Run sot_sequence
    for (int32_t i = 0; i < static_cast<int32_t>(sot_sequence.size()); ++i) {
      logits = RunDecoder(sot_sequence[i], offset, cross_kv, self_kv_data, mask);
      UpdateSelfKvCache(self_kv_data, offset);
      mask[offset] = 0;
      offset += 1;
    }

    int32_t idx = MaxElementIndex(logits.data(), logits.size());
    if (idx == eot_) {
      return r;
    }

    r.tokens.reserve(num_possible_tokens);

    while (offset < num_possible_tokens && idx != eot_) {
      r.tokens.push_back(idx);

      logits = RunDecoder(idx, offset, cross_kv, self_kv_data, mask);
      UpdateSelfKvCache(self_kv_data, offset);
      mask[offset] = 0;
      offset += 1;

      idx = MaxElementIndex(logits.data(), logits.size());
    }

    return r;
  }

  int32_t FeatureDim() const { return feat_dim_; }

 private:
  void InitEncoder() {
    const auto &qnn_config = config_.whisper.qnn_config;

    encoder_backend_ = std::make_unique<QnnBackend>(qnn_config.backend_lib,
                                                    config_.debug);

    const auto &context_binary = qnn_config.context_binary;

    std::vector<std::string> binary_filenames;
    SplitStringToVector(context_binary, ",", true, &binary_filenames);

    std::string encoder_binary;
    if (binary_filenames.size() >= 1) {
      encoder_binary = binary_filenames[0];
    }

    if (encoder_binary.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init encoder from model lib since context binary is not given");
      }

      InitEncoderFromModelLib();

      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Skip generating context binary since you don't provide a path to "
            "save it");
      }
    } else if (!FileExists(encoder_binary)) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init encoder from model lib since context binary '%s' does not "
            "exist",
            encoder_binary.c_str());
      }

      InitEncoderFromModelLib();

      CreateContextBinary(encoder_model_.get(), encoder_binary);
    } else {
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Init encoder from context binary '%s'",
                         encoder_binary.c_str());
      }
      InitEncoderFromContextBinary(encoder_binary);
    }
  }

  void InitDecoder() {
    const auto &qnn_config = config_.whisper.qnn_config;

    decoder_backend_ = std::make_unique<QnnBackend>(qnn_config.backend_lib,
                                                    config_.debug);

    const auto &context_binary = qnn_config.context_binary;

    std::vector<std::string> binary_filenames;
    SplitStringToVector(context_binary, ",", true, &binary_filenames);

    std::string decoder_binary;
    if (binary_filenames.size() >= 2) {
      decoder_binary = binary_filenames[1];
    }

    if (decoder_binary.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init decoder from model lib since context binary is not given");
      }

      InitDecoderFromModelLib();

      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Skip generating context binary since you don't provide a path to "
            "save it");
      }
    } else if (!FileExists(decoder_binary)) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init decoder from model lib since context binary '%s' does not "
            "exist",
            decoder_binary.c_str());
      }

      InitDecoderFromModelLib();

      CreateContextBinary(decoder_model_.get(), decoder_binary);
    } else {
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Init decoder from context binary '%s'",
                         decoder_binary.c_str());
      }
      InitDecoderFromContextBinary(decoder_binary);
    }
  }

  void InitEncoderFromModelLib() {
    const auto &encoder_path = config_.whisper.encoder;
    if (encoder_path.empty()) {
      SHERPA_ONNX_LOGE("Please provide --whisper-encoder");
      SHERPA_ONNX_EXIT(-1);
    }

    encoder_backend_->InitContext();
    encoder_model_ = std::make_unique<QnnModel>(encoder_path,
                                                encoder_backend_.get(),
                                                config_.debug);
  }

  void InitEncoderFromContextBinary(const std::string &context_binary) {
    const auto &system_lib = config_.whisper.qnn_config.system_lib;
    if (system_lib.empty()) {
      SHERPA_ONNX_LOGE(
          "You should provide --whisper.qnn-system-lib if you also provide "
          "context binary");
      SHERPA_ONNX_EXIT(-1);
    }

    encoder_model_ = std::make_unique<QnnModel>(context_binary, system_lib,
                                                encoder_backend_.get(),
                                                BinaryContextTag{},
                                                config_.debug);
  }

  void InitDecoderFromModelLib() {
    const auto &decoder_path = config_.whisper.decoder;
    if (decoder_path.empty()) {
      SHERPA_ONNX_LOGE("Please provide --whisper-decoder");
      SHERPA_ONNX_EXIT(-1);
    }

    decoder_backend_->InitContext();
    decoder_model_ = std::make_unique<QnnModel>(decoder_path,
                                                decoder_backend_.get(),
                                                config_.debug);
  }

  void InitDecoderFromContextBinary(const std::string &context_binary) {
    const auto &system_lib = config_.whisper.qnn_config.system_lib;
    if (system_lib.empty()) {
      SHERPA_ONNX_LOGE(
          "You should provide --whisper.qnn-system-lib if you also provide "
          "context binary");
      SHERPA_ONNX_EXIT(-1);
    }

    decoder_model_ = std::make_unique<QnnModel>(context_binary, system_lib,
                                                decoder_backend_.get(),
                                                BinaryContextTag{},
                                                config_.debug);
  }

  void CreateContextBinary(QnnModel *model,
                           const std::string &context_binary) {
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

  void PostInit() {
    PostInitEncoder();
    PostInitDecoder();
    InitSotSequence();
    PreComputeTensorNames();
  }

  void PostInitEncoder() {
    const std::vector<std::string> &names = encoder_model_->InputTensorNames();
    if (names.empty()) {
      SHERPA_ONNX_LOGE("Encoder has no input tensors");
      SHERPA_ONNX_EXIT(-1);
    }

    prefix_ = ParseModelPrefix(names[0]);
    model_type_ = ParseWhisperModelFromPrefix(prefix_);

    if (config_.debug) {
      SHERPA_ONNX_LOGE("model type: %s, prefix: %s",
                       ToString(model_type_).c_str(), prefix_.c_str());
    }

    std::string mel_name = prefix_ + "_mel";
    if (!encoder_model_->HasTensor(mel_name)) {
      SHERPA_ONNX_LOGE("Encoder does not have input tensor '%s'",
                       mel_name.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int32_t> mel_shape = encoder_model_->TensorShape(mel_name);
    if (mel_shape.size() != 3 || mel_shape[0] != 1) {
      SHERPA_ONNX_LOGE(
          "Encoder mel input should be 3-d with batch 1. Given shape: [%d, %d, "
          "%d]",
          mel_shape[0],
          mel_shape.size() > 1 ? mel_shape[1] : 0,
          mel_shape.size() > 2 ? mel_shape[2] : 0);
      SHERPA_ONNX_EXIT(-1);
    }

    num_frames_ = mel_shape[1];
    feat_dim_ = mel_shape[2];

    const std::vector<std::string> &output_names =
        encoder_model_->OutputTensorNames();
    n_text_layer_ = output_names.size() / 2;

    // Encoder outputs don't have prefix
    if (!encoder_model_->HasTensor("cross_k_0")) {
      SHERPA_ONNX_LOGE("Encoder does not have output tensor 'cross_k_0'");
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int32_t> cross_k_shape =
        encoder_model_->TensorShape("cross_k_0");

    num_out_frames_ = cross_k_shape[1];
    n_text_state_ = cross_k_shape[2];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("feat_dim_: %d", feat_dim_);
      SHERPA_ONNX_LOGE("num_frames_: %d", num_frames_);
      SHERPA_ONNX_LOGE("num_out_frames_: %d", num_out_frames_);
      SHERPA_ONNX_LOGE("n_text_layer_: %d", n_text_layer_);
      SHERPA_ONNX_LOGE("n_text_state_: %d", n_text_state_);
    }
  }

  void PostInitDecoder() {
    const std::vector<std::string> &input_names =
        decoder_model_->InputTensorNames();

    int32_t expected_num_inputs = 1 + 2 * n_text_layer_ + 2 * n_text_layer_ + 2;
    if (static_cast<int32_t>(input_names.size()) != expected_num_inputs) {
      SHERPA_ONNX_LOGE("Expect %d decoder inputs. Actual: %d",
                       expected_num_inputs,
                       static_cast<int32_t>(input_names.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    std::string mask_name = prefix_ + "_mask";
    if (!decoder_model_->HasTensor(mask_name)) {
      SHERPA_ONNX_LOGE("Decoder does not have input tensor '%s'",
                       mask_name.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int32_t> mask_shape = decoder_model_->TensorShape(mask_name);
    mask_size_ = mask_shape[0];

    std::string logits_name = prefix_ + "_logits";
    if (!decoder_model_->HasTensor(logits_name)) {
      SHERPA_ONNX_LOGE("Decoder does not have output tensor '%s'",
                       logits_name.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int32_t> logits_shape =
        decoder_model_->TensorShape(logits_name);
    vocab_size_ = logits_shape.back();

    // self kv cache size: mask_size * n_text_state
    self_kv_size_ = mask_size_ * n_text_state_;

    // The QNN model has swapped dimensions compared to ONNX:
    // ONNX self_k: (1, mask_size, n_text_state) → stride = n_text_state
    // QNN  self_k: (1, n_text_state, mask_size) → stride = mask_size
    self_kv_stride_ = mask_size_;

    if (config_.debug) {
      SHERPA_ONNX_LOGE("mask_size_: %d", mask_size_);
      SHERPA_ONNX_LOGE("n_text_state_: %d", n_text_state_);
      SHERPA_ONNX_LOGE("vocab_size_: %d", vocab_size_);
      SHERPA_ONNX_LOGE("self_kv_size_: %d", self_kv_size_);
      SHERPA_ONNX_LOGE("self_kv_stride_: %d", self_kv_stride_);
    }
  }

  void PreComputeTensorNames() {
    // Pre-compute tensor names to avoid string concatenation in hot loop
    mel_name_ = prefix_ + "_mel";
    tokens_name_ = prefix_ + "_tokens";
    offset_name_ = prefix_ + "_offset";
    mask_name_ = prefix_ + "_mask";
    logits_name_ = prefix_ + "_logits";

    encoder_cross_k_.resize(n_text_layer_);
    encoder_cross_v_.resize(n_text_layer_);
    decoder_cross_k_.resize(n_text_layer_);
    decoder_cross_v_.resize(n_text_layer_);
    decoder_self_k_.resize(n_text_layer_);
    decoder_self_v_.resize(n_text_layer_);
    decoder_this_self_k_.resize(n_text_layer_);
    decoder_this_self_v_.resize(n_text_layer_);

    for (int32_t i = 0; i < n_text_layer_; ++i) {
      std::string index = std::to_string(i);

      // Encoder outputs (no prefix)
      encoder_cross_k_[i] = "cross_k_" + index;
      encoder_cross_v_[i] = "cross_v_" + index;

      // Decoder inputs (with prefix)
      decoder_cross_k_[i] = prefix_ + "_cross_k_" + index;
      decoder_cross_v_[i] = prefix_ + "_cross_v_" + index;
      decoder_self_k_[i] = prefix_ + "_self_k_" + index;
      decoder_self_v_[i] = prefix_ + "_self_v_" + index;

      // Decoder outputs (with prefix)
      decoder_this_self_k_[i] = prefix_ + "_this_self_k_" + index;
      decoder_this_self_v_[i] = prefix_ + "_this_self_v_" + index;
    }
  }

  void InitSotSequence() {
    switch (model_type_) {
      case WhisperModelType::TinyEn:
      case WhisperModelType::BaseEn:
      case WhisperModelType::SmallEn:
      case WhisperModelType::MediumEn:
        sot_sequence_ = {50257, 50362};
        eot_ = 50256;
        break;
      case WhisperModelType::Tiny:
      case WhisperModelType::Base:
      case WhisperModelType::Small:
      case WhisperModelType::Medium:
      case WhisperModelType::Large:
        sot_sequence_ = {50258, 50259, 50359, 50363};
        eot_ = 50257;
        translate_ = 50358;
        break;
      default:
        SHERPA_ONNX_LOGE("Unsupported model type: '%s'",
                         ToString(model_type_).c_str());
        SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      std::ostringstream os;
      os << "sot_sequence: ";
      for (auto i : sot_sequence_) {
        os << i << " ";
      }
      os << "\n";
      os << "eot: " << eot_ << "\n";
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }
  }

  void RunEncoder(const std::vector<float> &features) const {
    encoder_model_->SetInputTensorData(mel_name_, features.data(),
                                       features.size());
    encoder_model_->Run();
  }

  std::vector<float> RunDecoder(
      int32_t token, int32_t offset,
      const std::vector<std::vector<float>> &cross_kv,
      const float *self_kv_data,
      const std::vector<int32_t> &mask) const {
    // Set tokens
    decoder_model_->SetInputTensorData(tokens_name_, &token, 1);

    // Set self kv cache (flat vector layout)
    for (int32_t i = 0; i < n_text_layer_; ++i) {
      decoder_model_->SetInputTensorData(decoder_self_k_[i],
                                         self_kv_data + (i * 2) * self_kv_size_,
                                         self_kv_size_);

      decoder_model_->SetInputTensorData(decoder_self_v_[i],
                                         self_kv_data + (i * 2 + 1) * self_kv_size_,
                                         self_kv_size_);
    }

    // Set cross kv cache (from encoder, transposed)
    for (int32_t i = 0; i < n_text_layer_; ++i) {
      decoder_model_->SetInputTensorData(decoder_cross_k_[i],
                                         cross_kv[i * 2].data(),
                                         cross_kv[i * 2].size());

      decoder_model_->SetInputTensorData(decoder_cross_v_[i],
                                         cross_kv[i * 2 + 1].data(),
                                         cross_kv[i * 2 + 1].size());
    }

    // Set offset
    decoder_model_->SetInputTensorData(offset_name_, &offset, 1);

    // Set mask
    decoder_model_->SetInputTensorData(mask_name_, mask.data(), mask.size());

    // Run decoder
    decoder_model_->Run();

    // Get logits
    return decoder_model_->GetOutputTensorData(logits_name_);
  }

  void UpdateSelfKvCache(float *self_kv_data, int32_t offset) const {
    for (int32_t i = 0; i < n_text_layer_; ++i) {
      // Update self_k
      auto delta_k =
          decoder_model_->GetOutputTensorData(decoder_this_self_k_[i]);
      float *self_k = self_kv_data + (i * 2) * self_kv_size_;
      for (size_t r = 0; r != delta_k.size(); ++r) {
        self_k[r * self_kv_stride_ + offset] += delta_k[r];
      }

      // Update self_v
      auto delta_v =
          decoder_model_->GetOutputTensorData(decoder_this_self_v_[i]);
      float *self_v = self_kv_data + (i * 2 + 1) * self_kv_size_;
      for (size_t r = 0; r != delta_v.size(); ++r) {
        self_v[r * self_kv_stride_ + offset] += delta_v[r];
      }
    }
  }

  int32_t DetectLanguage(
      const std::vector<std::vector<float>> &cross_kv,
      std::vector<float> &self_kv,
      const std::vector<int32_t> &mask) const {
    int32_t offset = 0;
    auto logits =
        RunDecoder(sot_sequence_[0], offset, cross_kv, self_kv.data(), mask);

    const auto &all_lang_ids = GetAllWhisperLanguageTokenIds();
    int32_t lang_id = all_lang_ids[0];
    float this_logit = logits[lang_id];

    for (int32_t i = 1; i != static_cast<int32_t>(all_lang_ids.size()); ++i) {
      int32_t id = all_lang_ids[i];
      float p = logits[id];

      if (p > this_logit) {
        this_logit = p;
        lang_id = id;
      }
    }

    return lang_id;
  }

 private:
  mutable std::mutex mutex_;
  OfflineModelConfig config_;

  std::unique_ptr<QnnBackend> encoder_backend_;
  std::unique_ptr<QnnModel> encoder_model_;

  std::unique_ptr<QnnBackend> decoder_backend_;
  std::unique_ptr<QnnModel> decoder_model_;

  // Model prefix (e.g., "tiny_en")
  std::string prefix_;

  // Model type
  WhisperModelType model_type_;

  // Dimensions
  int32_t feat_dim_ = 0;
  int32_t num_frames_ = 0;
  int32_t num_out_frames_ = 0;
  int32_t n_text_layer_ = 0;
  int32_t n_text_state_ = 0;
  int32_t vocab_size_ = 0;
  int32_t mask_size_ = 0;
  int32_t self_kv_size_ = 0;
  int32_t self_kv_stride_ = 0;

  // Pre-computed tensor names (avoid string concatenation in hot loop)
  std::string mel_name_;
  std::string tokens_name_;
  std::string offset_name_;
  std::string mask_name_;
  std::string logits_name_;

  std::vector<std::string> encoder_cross_k_;
  std::vector<std::string> encoder_cross_v_;
  std::vector<std::string> decoder_cross_k_;
  std::vector<std::string> decoder_cross_v_;
  std::vector<std::string> decoder_self_k_;
  std::vector<std::string> decoder_self_v_;
  std::vector<std::string> decoder_this_self_k_;
  std::vector<std::string> decoder_this_self_v_;

  // SOT sequence
  std::vector<int32_t> sot_sequence_;
  int32_t eot_ = 0;
  int32_t translate_ = 0;
};

OfflineWhisperModelQnn::OfflineWhisperModelQnn(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineWhisperModelQnn::OfflineWhisperModelQnn(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineWhisperModelQnn::~OfflineWhisperModelQnn() = default;

OfflineWhisperDecoderResult OfflineWhisperModelQnn::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

int32_t OfflineWhisperModelQnn::FeatureDim() const {
  return impl_->FeatureDim();
}

#if __ANDROID_API__ >= 9
template OfflineWhisperModelQnn::OfflineWhisperModelQnn(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineWhisperModelQnn::OfflineWhisperModelQnn(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
