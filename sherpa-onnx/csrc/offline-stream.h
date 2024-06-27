// sherpa-onnx/csrc/offline-stream.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
#define SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineRecognitionResult {
  // Recognition results.
  // For English, it consists of space separated words.
  // For Chinese, it consists of Chinese words without spaces.
  std::string text;

  // Decoded results at the token level.
  // For instance, for BPE-based models it consists of a list of BPE tokens.
  std::vector<std::string> tokens;

  /// timestamps.size() == tokens.size()
  /// timestamps[i] records the time in seconds when tokens[i] is decoded.
  ///
  /// Note: It is the start time stamp of a token.
  ///
  /// It is empty if the model does not support time stamp information.
  std::vector<float> timestamps;

  /// It is not empty for CTC models.
  /// It is empty for non-CTC models.
  /// If it is not empty, then stop_timestamps.size() == timestamps.size()
  std::vector<float> stop_timestamps;

  /// It is not empty for CTC models with a HL or HLG decoding graph
  /// It is empty for non-CTC models.
  ///
  /// If not empty, it contains word IDs. You have to use words.txt
  /// to map word IDs to word symbols.
  std::vector<int32_t> words;

  /// If not empty, word_start_timestamps[i] is the start time of words[i].
  ///
  /// words.size() == word_start_timestamps.size()
  std::vector<float> word_start_timestamps;

  std::string AsJsonString() const;
};

struct WhisperTag {};
struct CEDTag {};

class OfflineStream {
 public:
  explicit OfflineStream(const FeatureExtractorConfig &config = {},
                         ContextGraphPtr context_graph = {});

  explicit OfflineStream(WhisperTag tag);
  explicit OfflineStream(CEDTag tag);
  ~OfflineStream();

  /**
     @param sampling_rate The sampling_rate of the input waveform. If it does
                          not equal to  config.sampling_rate, we will do
                          resampling inside.
     @param waveform Pointer to a 1-D array of size n. It must be normalized to
                     the range [-1, 1].
     @param n Number of entries in waveform

     Caution: You can only invoke this function once so you have to input
              all the samples at once
   */
  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const;

  /// Return feature dim of this extractor
  int32_t FeatureDim() const;

  // Get all the feature frames of this stream in a 1-D array, which is
  // flattened from a 2-D array of shape (num_frames, feat_dim).
  std::vector<float> GetFrames() const;

  /** Set the recognition result for this stream. */
  void SetResult(const OfflineRecognitionResult &r);

  /** Get the recognition result of this stream */
  const OfflineRecognitionResult &GetResult() const;

  /** Get the ContextGraph of this stream */
  const ContextGraphPtr &GetContextGraph() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
