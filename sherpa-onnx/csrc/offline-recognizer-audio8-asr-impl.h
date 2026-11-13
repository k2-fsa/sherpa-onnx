// sherpa-onnx/csrc/offline-recognizer-audio8-asr-impl.h
//
// Copyright (c)  2026  yuyun2000

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_AUDIO8_ASR_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_AUDIO8_ASR_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-audio8-asr-model.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/qwen-asr-tokenizer.h"

namespace sherpa_onnx {

// Audio8-ASR-0.1B: a Whisper-style audio encoder plus a small Qwen2 decoder.
class OfflineRecognizerAudio8AsrImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerAudio8AsrImpl(
      const OfflineRecognizerConfig &config);

  template <typename Manager>
  OfflineRecognizerAudio8AsrImpl(Manager *mgr,
                                 const OfflineRecognizerConfig &config);

  std::unique_ptr<OfflineStream> CreateStream() const override;

  void DecodeStreams(OfflineStream **ss, int32_t n) const override;

  void Decode(OfflineStream *stream) const;

  OfflineRecognizerConfig GetConfig() const override { return config_; }

  void SetConfig(const OfflineRecognizerConfig &config) override {
    config_ = config;
  }

 private:
  void InitPromptIds();

  OfflineRecognitionResult GenerateText(const OfflineStream *stream,
                                        std::vector<float> *audio_features,
                                        int32_t num_audio_tokens,
                                        int32_t audio_dim) const;

  int64_t SampleToken(const float *logits, int32_t vocab_size) const;

 private:
  OfflineRecognizerConfig config_;
  std::unique_ptr<OfflineAudio8AsrModel> model_;
  std::unique_ptr<QwenAsrTokenizer> tokenizer_;

  int32_t audio_token_id_ = -1;
  std::vector<int64_t> prompt_prefix_ids_;   // <|user|><|begin_of_audio|>
  std::vector<int64_t> prompt_suffix_ids_;   // <|end_of_audio|>Please...
  int32_t eos_id_ = -1;
  int32_t pad_id_ = -1;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_AUDIO8_ASR_IMPL_H_
