// sherpa-onnx/csrc/spoken-language-identification-whisper-impl.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_WHISPER_IMPL_H_
#define SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_WHISPER_IMPL_H_

namespace sherpa_onnx {

class SpokenLanguageIdentificationWhisperImpl
    : public SpokenLanguageIdentificationImpl {
 public:
  explicit SpokenLanguageIdentificationWhisperImpl(
      const SpokenLanguageIdentificationConfig &config) {}

  std::unique_ptr<OnlineStream> CreateStream() const override {
    return nullptr;
  }

  bool IsReady(OnlineStream *s) const override { return true; }

  std::string Compute(OnlineStream *s) const override { return ""; }
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_WHISPER_IMPL_H_
