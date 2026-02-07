// sherpa-onnx/csrc/offline-tts-supertonic-impl.h
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-supertonic-model.h"
#include "sherpa-onnx/csrc/offline-tts-supertonic-unicode-processor.h"

namespace sherpa_onnx {

class OfflineTtsSupertonicImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsSupertonicImpl(const OfflineTtsConfig &config);

  template <typename Manager>
  OfflineTtsSupertonicImpl(Manager *mgr, const OfflineTtsConfig &config);

  int32_t SampleRate() const override;

  int32_t NumSpeakers() const override { return 0; }

  GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override;

  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &config,
      GeneratedAudioCallback callback = nullptr) const override;

 private:
  GeneratedAudio Process(const std::vector<std::string> &text_list,
                         const std::vector<std::string> &lang_list,
                         const SupertonicStyle &style, int32_t num_steps,
                         float speed) const;

  GeneratedAudio ProcessChunksAndConcatenate(
      const std::vector<std::string> &text_chunks, const std::string &lang,
      const SupertonicStyle &style, int32_t num_steps, float speed,
      float silence_duration, GeneratedAudioCallback callback) const;

  SupertonicStyle LoadVoiceStyle(const std::string &voice_style_path) const;

  SupertonicStyle LoadVoiceStyles(
      const std::vector<std::string> &voice_style_paths) const;

  template <typename Manager>
  SupertonicStyle LoadVoiceStyle(Manager *mgr,
                                 const std::string &voice_style_path) const;

  template <typename Manager>
  SupertonicStyle LoadVoiceStyles(
      Manager *mgr, const std::vector<std::string> &voice_style_paths) const;

  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsSupertonicModel> model_;
  std::unique_ptr<SupertonicUnicodeProcessor> text_processor_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_IMPL_H_
