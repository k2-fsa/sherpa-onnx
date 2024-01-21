// sherpa-onnx/csrc/speaker-embedding-extractor.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_H_

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct SpeakerEmbeddingExtractorConfig {
  std::string model;
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  SpeakerEmbeddingExtractorConfig() = default;
  SpeakerEmbeddingExtractorConfig(const std::string &model, int32_t num_threads,
                                  bool debug, const std::string &provider)
      : model(model),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

class SpeakerEmbeddingExtractorImpl;

class SpeakerEmbeddingExtractor {
 public:
  explicit SpeakerEmbeddingExtractor(
      const SpeakerEmbeddingExtractorConfig &config);

#if __ANDROID_API__ >= 9
  SpeakerEmbeddingExtractor(AAssetManager *mgr,
                            const SpeakerEmbeddingExtractorConfig &config);
#endif

  ~SpeakerEmbeddingExtractor();

  // Return the dimension of the embedding
  int32_t Dim() const;

  // Create a stream to accept audio samples and compute features
  std::unique_ptr<OnlineStream> CreateStream() const;

  // Return true if there are feature frames in OnlineStream that
  // can be used to compute embeddings.
  bool IsReady(OnlineStream *s) const;

  // Compute the speaker embedding from the available unprocessed features
  // of the given stream
  //
  // You have to ensure IsReady(s) returns true before you call this method.
  std::vector<float> Compute(OnlineStream *s) const;

 private:
  std::unique_ptr<SpeakerEmbeddingExtractorImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_H_
