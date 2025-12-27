// sherpa-onnx/csrc/offline-whisper-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_CONFIG_H_

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineWhisperModelConfig {
  std::string encoder;
  std::string decoder;

  // Available languages can be found at
  // https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
  //
  // Note: For non-multilingual models, it supports only "en"
  //
  // If empty, we will infer it from the input audio file when
  // the model is multilingual.
  std::string language;

  // Valid values are transcribe and translate
  //
  // Note: For non-multilingual models, it supports only "transcribe"
  std::string task = "transcribe";

  // Number of tail padding frames.
  //
  // Since we remove the 30-second constraint, we need to add some paddings
  // at the end.
  //
  // Recommended values:
  //   - 50 for English models
  //   - 300 for multilingual models
  int32_t tail_paddings = -1;

  // If true, use cross-attention weights and DTW to compute token-level
  // timestamps. This requires ONNX models exported with attention outputs.
  bool enable_timestamps = false;

  OfflineWhisperModelConfig() = default;
  OfflineWhisperModelConfig(const std::string &encoder,
                            const std::string &decoder,
                            const std::string &language,
                            const std::string &task, int32_t tail_paddings,
                            bool enable_timestamps = false)
      : encoder(encoder),
        decoder(decoder),
        language(language),
        task(task),
        tail_paddings(tail_paddings),
        enable_timestamps(enable_timestamps) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct OfflineWhisperDecoderResult {
  /// The decoded token IDs
  std::vector<int32_t> tokens;
  std::string lang;

  /// Cross-attention weights for token-level timestamps (if enabled)
  /// Shape: (n_heads, n_tokens, n_audio_frames), flattened to 1D
  /// Empty if timestamps are not enabled or model doesn't support it
  std::vector<float> attention_weights;

  /// Dimensions of attention weights
  int32_t attention_n_heads = 0;
  int32_t attention_n_tokens = 0;
  int32_t attention_n_frames = 0;

  /// Number of actual audio feature frames (for clipping attention)
  /// This is num_feature_frames / 2 (due to encoder downsampling)
  int32_t num_audio_frames = 0;
};

// used by ascend/rknn/qnn/axera, etc.
enum class WhisperModelType {
  Tiny,
  TinyEn,
  Base,
  BaseEn,
  Small,
  SmallEn,
  Medium,
  MediumEn,
  Large
};

std::string ToString(WhisperModelType model);
bool IsMultilingual(WhisperModelType model_type);

WhisperModelType ParseWhisperModelType(const std::string &name);
int32_t GetWhisperLanguageTokenId(const std::string &lang);
std::string GetWhisperLanguageCode(int32_t token_id);
const std::vector<int32_t> &GetAllWhisperLanguageTokenIds();
const std::vector<std::string> &GetAllWhisperLanguageCodes();

struct WhisperModelMultilingualTokens {
  int32_t sot = 50258;
  int32_t eot = 50257;
  int32_t transcribe = 50359;
  int32_t translate = 50358;
  int32_t no_timestamps = 50363;
};

struct WhisperModelEnglishTokens {
  int32_t sot = 50257;
  int32_t eot = 50256;
  int32_t no_timestamps = 50362;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_CONFIG_H_
