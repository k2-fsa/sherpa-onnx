// sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h
//
// Copyright (c)  2026   zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_QWEN3_ASR_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_QWEN3_ASR_IMPL_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-qwen3-asr-model.h"
#include "sherpa-onnx/csrc/offline-qwen3-forced-aligner-model.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/qwen-asr-tokenizer.h"

namespace sherpa_onnx {

// Trims trailing near-silent frames from audio_features (shape [1, A, H]).
// If every frame's energy stays below the silence threshold, the tensor is
// returned unmodified and, when |all_silent| is not null, |*all_silent| is
// set to true so the caller can treat the clip as having no speech content.
// Exposed here (rather than kept file-local) so it can be unit tested.
Ort::Value TrimAudioFeatures(Ort::Value audio_features, OrtAllocator *allocator,
                             bool *all_silent = nullptr);

// Splits UTF-8 text into word units for the Qwen3 forced aligner, mirroring
// tokenize_space_lang() in qwen3_forced_aligner.py: split on whitespace,
// drop characters that are not letters/numbers/'', then emit each CJK
// character as its own unit. Exposed for unit testing.
std::vector<std::string> SplitQwen3AlignerWords(const std::string &text);

// Applies the longest-increasing-subsequence monotonicity fix from
// fix_timestamp() in qwen3_forced_aligner.py. |data| holds raw timestamp
// indices (in 80 ms units) and is modified in place. Exposed for unit
// testing.
void FixQwen3AlignerTimestamps(std::vector<int64_t> *data);

class OfflineRecognizerQwen3ASRImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerQwen3ASRImpl(const OfflineRecognizerConfig &config);

  template <typename Manager>
  OfflineRecognizerQwen3ASRImpl(Manager *mgr,
                                const OfflineRecognizerConfig &config);

  std::unique_ptr<OfflineStream> CreateStream() const override;

  void DecodeStreams(OfflineStream **ss, int32_t n) const override;

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void InitPromptTemplateIds();
  std::vector<int64_t> BuildSourceIds(const std::string &hotwords,
                                      const std::string &language,
                                      int32_t audio_token_len,
                                      int32_t *before_len,
                                      int32_t *fake_audio_token_len) const;

  int64_t SampleTokenFromLogitsFp16OrFp32(const void *logits, bool is_fp16,
                                          int32_t vocab_size) const;
  int64_t SampleTokenFromLogits(const Ort::Value &logits, int32_t time_index,
                                float temperature, float top_p) const;

  int64_t SampleTokenWithTemperatureAndTopP(const void *logits, bool is_fp16,
                                            int32_t vocab_size,
                                            float temperature, float top_p,
                                            int64_t avoid_id = -1) const;

  OfflineRecognitionResult GenerateText(Ort::Value audio_features,
                                        int32_t audio_token_len,
                                        OfflineStream *stream) const;

  // Runs the optional Qwen3-ForcedAligner over the same mel features and the
  // decoded text, then fills r->tokens/timestamps/durations with word-level
  // results. Returns false (leaving |r| untouched) when alignment is not
  // applicable.
  bool RunForcedAlignment(const std::vector<float> &mel_features,
                          int32_t feat_frames,
                          OfflineRecognitionResult *r) const;

  void Decode(OfflineStream *stream) const;

  OfflineRecognizerConfig config_;
  std::unique_ptr<OfflineQwen3ASRModel> model_;
  std::unique_ptr<OfflineQwen3ForcedAlignerModel> aligner_model_;
  std::unique_ptr<QwenAsrTokenizer> tokenizer_;
  // Separate tokenizer for the forced aligner: its vocab contains the extra
  // <timestamp> slot token that the ASR vocab lacks.
  std::unique_ptr<QwenAsrTokenizer> aligner_tokenizer_;
  std::vector<int64_t> audio_pad_ids_;
  std::vector<int64_t> prompt_ids_after_;
  int64_t asr_text_token_id_ = -1;
  int64_t aligner_audio_start_token_id_ = -1;
  int64_t aligner_audio_end_token_id_ = -1;
  int64_t aligner_audio_pad_token_id_ = -1;
  int64_t aligner_timestamp_token_id_ = -1;
  mutable std::mt19937 rng_;
  // Protects rng_, which is shared and drawn from by
  // SampleTokenWithTemperatureAndTopP(). DecodeStreams() may be called
  // concurrently from multiple threads on the same recognizer instance (see
  // sherpa-onnx-offline-parallel.cc), so draws from rng_ must be serialized to
  // avoid a data race.
  mutable std::mutex rng_mutex_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_QWEN3_ASR_IMPL_H_
