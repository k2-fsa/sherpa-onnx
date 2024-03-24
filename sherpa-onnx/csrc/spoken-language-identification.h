// sherpa-onnx/csrc/spoken-language-identification.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_H_
#define SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_H_

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct SpokenLanguageIdentificationConfig {
  // 1. For whisper models:
  // Requires a multi-lingual whisper encoder model.
  // That is, it supports only tiny, base, small, medium, large.
  // Note: It does NOT support tiny.en, base.en, small.en, medium.en
  // You only need to pass the encoder model of whisper.
  //
  // 2. It does not support non-whisper models at present.
  std::string model;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  SpokenLanguageIdentificationConfig() = default;

  SpokenLanguageIdentificationConfig(const std::string &model,
                                     int32_t num_threads, bool debug,
                                     const std::string &provider)
      : model(model),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

class SpokenLanguageIdentificationImpl;

class SpokenLanguageIdentification {
 public:
  explicit SpokenLanguageIdentification(
      const SpokenLanguageIdentificationConfig &config);

  ~SpokenLanguageIdentification();

  // Create a stream to accept audio samples and compute features
  std::unique_ptr<OnlineStream> CreateStream() const;

  // Return true if there are feature frames in OnlineStream that
  // can be used to compute embeddings.
  bool IsReady(OnlineStream *s) const;

  // Return a string containg the language, e.g., English, Chinese, German, etc.
  //
  // You have to ensure IsReady(s) returns true before you call this method.
  std::string Compute(OnlineStream *s) const;

 private:
  std::unique_ptr<SpokenLanguageIdentificationImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPOKEN_LANGUAGE_IDENTIFICATION_H_
