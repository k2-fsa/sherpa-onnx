// sherpa-onnx/csrc/axcl/offline-tts-supertonic-impl-axcl.h
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#ifndef SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_SUPERTONIC_IMPL_AXCL_H_
#define SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_SUPERTONIC_IMPL_AXCL_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/axcl/offline-tts-supertonic-model-axcl.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/normal-data-generator.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-supertonic-model.h"
#include "sherpa-onnx/csrc/offline-tts-supertonic-unicode-processor.h"

namespace sherpa_onnx {

class OfflineTtsSupertonicImplAxcl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsSupertonicImplAxcl(const OfflineTtsConfig &config);

  template <typename Manager>
  OfflineTtsSupertonicImplAxcl(Manager *mgr,
                                const OfflineTtsConfig &config);

  int32_t SampleRate() const override;

  int32_t NumSpeakers() const override { return num_speakers_; }

  [[deprecated("Use Generate(text, GenerationConfig, callback) instead")]]
  GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override;

  GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &config,
      GeneratedAudioCallback callback = nullptr) const override;

 private:
  GeneratedAudio Process(const std::string &text,
                         const std::string &lang, int64_t sid,
                         int32_t num_steps, float speed,
                         NormalDataGenerator &gen) const;

  GeneratedAudio ProcessChunksAndConcatenate(
      const std::vector<std::string> &text_chunks, const std::string &lang,
      int64_t sid, int32_t num_steps, float speed, float silence_duration,
      int32_t seed, GeneratedAudioCallback callback) const;

  void InitVoiceStyle(const std::vector<char> &buf);

  struct StyleSliceView {
    const float *ttl_data;
    size_t ttl_size;
    std::array<int64_t, 3> ttl_shape;
    const float *dp_data;
    size_t dp_size;
    std::array<int64_t, 3> dp_shape;
  };
  StyleSliceView GetStyleSliceForSid(int64_t sid) const;

  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsSupertonicModelAxcl> model_;
  std::unique_ptr<SupertonicUnicodeProcessor> text_processor_;
  int32_t num_speakers_ = 0;
  SupertonicStyle full_style_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_OFFLINE_TTS_SUPERTONIC_IMPL_AXCL_H_
