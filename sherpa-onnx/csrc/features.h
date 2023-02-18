// sherpa/csrc/features.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FEATURES_H_
#define SHERPA_ONNX_CSRC_FEATURES_H_

#include <memory>
#include <mutex>  // NOLINT
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace sherpa_onnx {

class FeatureExtractor {
 public:
  FeatureExtractor();
  explicit FeatureExtractor(const knf::FbankOptions &fbank_opts);

  /**
     @param sampling_rate The sampling_rate of the input waveform. Should match
                          the one expected by the feature extractor.
     @param waveform Pointer to a 1-D array of size n
     @param n Number of entries in waveform
   */
  void AcceptWaveform(float sampling_rate, const float *waveform, int32_t n);

  // InputFinished() tells the class you won't be providing any
  // more waveform.  This will help flush out the last frame or two
  // of features, in the case where snip-edges == false; it also
  // affects the return value of IsLastFrame().
  void InputFinished();

  int32_t NumFramesReady() const;

  // Note: IsLastFrame() will only ever return true if you have called
  // InputFinished() (and this frame is the last frame).
  bool IsLastFrame(int32_t frame) const;

  /** Get n frames starting from the given frame index.
   *
   * @param frame_index  The starting frame index
   * @param n  Number of frames to get.
   * @return Return a 2-D tensor of shape (n, feature_dim).
   *         which is flattened into a 1-D vector (flattened in in row major)
   */
  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const;

  void Reset();
  int32_t FeatureDim() const { return opts_.mel_opts.num_bins; }

 private:
  std::unique_ptr<knf::OnlineFbank> fbank_;
  knf::FbankOptions opts_;
  mutable std::mutex mutex_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FEATURES_H_
