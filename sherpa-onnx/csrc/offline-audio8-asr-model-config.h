// sherpa-onnx/csrc/offline-audio8-asr-model-config.h
//
// Copyright (c)  2026  yuyun2000

#ifndef SHERPA_ONNX_CSRC_OFFLINE_AUDIO8_ASR_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_AUDIO8_ASR_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// Configuration of Audio8-ASR-0.1B (an audio encoder + a small Qwen2 LM).
struct OfflineAudio8AsrModelConfig {
  // Path to encoder.onnx. It maps a 30 s Whisper log-mel window to audio
  // hidden states.
  std::string encoder;

  // Path to decoder.onnx. It contains the audio projector and the Qwen2 LM
  // with a fixed-size KV cache.
  std::string decoder;

  // Path to the directory holding tokenizer.json / vocab.json / merges.txt.
  std::string tokenizer;

  // Maximum number of generated tokens per utterance. 128 is enough for
  // roughly two minutes of speech with this tokenizer; the actual limit is
  // smaller when the prompt is long, because prompt + generated tokens must fit
  // into the KV cache of the decoder graph (see GetMaxTotalLen()). It can be
  // overridden per stream with stream->SetOption("max_new_tokens", "...");
  int32_t max_new_tokens = 128;

  OfflineAudio8AsrModelConfig() = default;

  OfflineAudio8AsrModelConfig(const std::string &encoder,
                              const std::string &decoder,
                              const std::string &tokenizer,
                              int32_t max_new_tokens)
      : encoder(encoder),
        decoder(decoder),
        tokenizer(tokenizer),
        max_new_tokens(max_new_tokens) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_AUDIO8_ASR_MODEL_CONFIG_H_
