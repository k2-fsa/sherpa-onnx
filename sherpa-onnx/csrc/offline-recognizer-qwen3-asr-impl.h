// sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h
//
// Copyright (c)  2026   zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_QWEN3_ASR_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_QWEN3_ASR_IMPL_H_

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-qwen3-asr-model.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/qwen-asr-tokenizer.h"

namespace sherpa_onnx {

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
  std::vector<int64_t> BuildSourceIds(int32_t audio_token_len,
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

  void Decode(OfflineStream *stream) const;

  OfflineRecognizerConfig config_;
  std::unique_ptr<OfflineQwen3ASRModel> model_;
  std::unique_ptr<QwenAsrTokenizer> tokenizer_;
  std::vector<int64_t> prompt_ids_before_;
  std::vector<int64_t> audio_pad_ids_;
  std::vector<int64_t> prompt_ids_after_;
  mutable std::mt19937 rng_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_QWEN3_ASR_IMPL_H_
