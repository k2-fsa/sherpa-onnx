// sherpa-onnx/csrc/audio-tagging.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AUDIO_TAGGING_H_
#define SHERPA_ONNX_CSRC_AUDIO_TAGGING_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {

struct AudioTaggingConfig {
  int32_t top_k = 5;
};

struct AudioEvent {
  std::string name;  // name of the event
  float prob;        // probability of the event
};

class AudioTaggingImpl;

class AudioTagging {
 public:
  explicit AudioTagging(const AudioTaggingConfig &config);

  ~AudioTagging();

  std::unique_ptr<OfflineStream> CreateStream() const;

  // If top_k is -1, then config.top_k is used.
  // Otherwise, config.top_k is ignored
  //
  // Return top_k AudioEvent. ans[0].prob is the largest of all returned events.
  std::vector<AudioEvent> Compute(OfflineStream *s, int32_t top_k = -1) const;

 private:
  std::unique_ptr<AudioTaggingImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_H_
