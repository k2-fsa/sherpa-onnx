// sherpa-onnx/csrc/online-stream.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_STREAM_H_
#define SHERPA_ONNX_CSRC_ONLINE_STREAM_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/online-ctc-decoder.h"
#include "sherpa-onnx/csrc/online-paraformer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"

namespace sherpa_onnx {

class OnlineStream {
 public:
  explicit OnlineStream(const FeatureExtractorConfig &config = {},
                        ContextGraphPtr context_graph = nullptr);
  ~OnlineStream();

  /**
     @param sampling_rate The sampling_rate of the input waveform. If it does
                          not equal to  config.sampling_rate, we will do
                          resampling inside.
     @param waveform Pointer to a 1-D array of size n. It must be normalized to
                     the range [-1, 1].
     @param n Number of entries in waveform
   */
  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const;

  /**
   * InputFinished() tells the class you won't be providing any
   * more waveform.  This will help flush out the last frame or two
   * of features, in the case where snip-edges == false; it also
   * affects the return value of IsLastFrame().
   */
  void InputFinished() const;

  int32_t NumFramesReady() const;

  /** Note: IsLastFrame() will only ever return true if you have called
   * InputFinished() (and this frame is the last frame).
   */
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

  int32_t FeatureDim() const;

  // Return a reference to the number of processed frames so far
  // before subsampling..
  // Initially, it is 0. It is always less than NumFramesReady().
  //
  // The returned reference is valid as long as this object is alive.
  int32_t &GetNumProcessedFrames();  // It's reset after calling Reset()

  int32_t GetNumFramesSinceStart() const;

  int32_t &GetCurrentSegment();

  void SetResult(const OnlineTransducerDecoderResult &r);
  OnlineTransducerDecoderResult &GetResult();

  void SetCtcResult(const OnlineCtcDecoderResult &r);
  OnlineCtcDecoderResult &GetCtcResult();

  void SetParaformerResult(const OnlineParaformerDecoderResult &r);
  OnlineParaformerDecoderResult &GetParaformerResult();

  void SetStates(std::vector<Ort::Value> states);
  std::vector<Ort::Value> &GetStates();

  /**
   * Get the context graph corresponding to this stream.
   *
   * @return Return the context graph for this stream.
   */
  const ContextGraphPtr &GetContextGraph() const;

  // for streaming parformer
  std::vector<float> &GetParaformerFeatCache();
  std::vector<float> &GetParaformerEncoderOutCache();
  std::vector<float> &GetParaformerAlphaCache();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_STREAM_H_
