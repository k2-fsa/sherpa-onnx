// sherpa-onnx/csrc/online-stream.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_STREAM_H_
#define SHERPA_ONNX_CSRC_ONLINE_STREAM_H_

#include <memory>
#include <string>
#include <vector>

#include "kaldi-decoder/csrc/faster-decoder.h"
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/online-ctc-decoder.h"
#include "sherpa-onnx/csrc/online-paraformer-decoder.h"
#include "sherpa-onnx/csrc/online-stream-state.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"

namespace sherpa_onnx {

struct TransducerKeywordResult;
class OnlineStream {
 public:
  explicit OnlineStream(const FeatureExtractorConfig &config = {},
                        ContextGraphPtr context_graph = nullptr);

  virtual ~OnlineStream();

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
   * @param frame_index  The starting frame index, relative to the last Reset().
   *                     It may be negative to read retained left context; the
   *                     caller must not request frames already discarded by
   *                     an earlier GetFrames() call.
   * @param n  Number of frames to get.
   * @return Return a 2-D tensor of shape (n, feature_dim).
   *         which is flattened into a 1-D vector (flattened in row major)
   */
  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const;

  // Internal buffered NeMo feature extraction; indices are segment-relative.
  std::vector<float> GetBufferedFeatures(int32_t frame_index, int32_t n,
                                         bool is_final,
                                         int32_t *start_sample_offset) const;

  int32_t NumTrailingSamples() const;

  void Reset();

  int32_t FeatureDim() const;

  // Return a reference to the number of processed frames so far
  // before subsampling..
  // Initially, it is 0. It is always less than NumFramesReady().
  //
  // The returned reference is valid as long as this object is alive.
  int32_t &GetNumProcessedFrames();  // It's reset after calling Reset()

  int32_t GetNumFramesSinceStart() const;

  // Process-unique stream id, for logging.
  int64_t Id() const;

  int32_t &GetCurrentSegment();

  void SetResult(const OnlineTransducerDecoderResult &r);
  OnlineTransducerDecoderResult &GetResult();

  void SetKeywordResult(const TransducerKeywordResult &r);
  TransducerKeywordResult &GetKeywordResult(bool remove_duplicates = false);

  void SetCtcResult(const OnlineCtcDecoderResult &r);
  OnlineCtcDecoderResult &GetCtcResult();

  void SetParaformerResult(const OnlineParaformerDecoderResult &r);
  OnlineParaformerDecoderResult &GetParaformerResult();

  void SetStates(std::vector<Ort::Value> states);
  std::vector<Ort::Value> &GetStates();

  void SetQnnStates(std::vector<OnlineStreamStateTensor> states);
  std::vector<OnlineStreamStateTensor> &GetQnnStates();

  void SetQnnResult(const OnlineTransducerDecoderResultNoOrt &r);
  OnlineTransducerDecoderResultNoOrt &GetQnnResult();

  void SetNeMoDecoderStates(std::vector<Ort::Value> decoder_states);
  void SetNeMoDecoderOut(Ort::Value decoder_out);

  std::vector<Ort::Value> &GetNeMoDecoderStates();
  Ort::Value &GetNeMoDecoderOut();

  /**
   * Get the context graph corresponding to this stream.
   *
   * @return Return the context graph for this stream.
   */
  const ContextGraphPtr &GetContextGraph() const;

  // for online ctc decoder
  void SetFasterDecoder(std::unique_ptr<kaldi_decoder::FasterDecoder> decoder);
  kaldi_decoder::FasterDecoder *GetFasterDecoder() const;
  int32_t &GetFasterDecoderProcessedFrames();

  // for streaming paraformer
  std::vector<float> &GetParaformerFeatCache();
  std::vector<float> &GetParaformerEncoderOutCache();
  std::vector<float> &GetParaformerAlphaCache();

  // Generic per-stream option mechanism (key-value string pairs).
  void SetOption(const std::string &key, const std::string &value);
  bool HasOption(const std::string &key) const;

  // Returns the value for the given key, or an empty string if the key
  // does not exist. No exception is thrown for missing keys.
  const std::string &GetOption(const std::string &key) const;
  int32_t GetOptionInt(const std::string &key, int32_t default_value = 0) const;
  float GetOptionFloat(const std::string &key,
                       float default_value = 0.0f) const;

  // ---- Independent VAD pipeline (ds-k2-sherpa-onnx refactor_vad style) ----
  //
  // Unlike a design where VAD runs inline inside AcceptWaveform(), here
  // AcceptWaveform() only appends a *copy* of the incoming waveform to a
  // separate raw-audio buffer; actual Silero inference only happens when
  // something calls VadProcessStep() explicitly (the websocket server
  // does this from its own, faster-cadence timer, running ahead of the
  // ASR decode loop -- see OnlineRecognizer::VadProcess()). This mirrors
  // the source architecture's OnlineVadModel/OnlineVadBuffer split: a
  // prediction buffer the ASR side polls for coverage, written by a VAD
  // step the ASR side never calls itself.

  // Enables VAD on this stream; the model is shared, the stream keeps its
  // own state. `window_shift`/`window_size` are in samples at
  // `vad_sample_rate`. `flush_supported`: the decoder can run the
  // end-of-utterance flush (see IsVadFlushPending()).
  void InitVad(int32_t vad_sample_rate, int32_t window_shift,
               int32_t window_size, float asr_frame_shift_ms,
               bool flush_supported);

  // True if InitVad() was called on this stream.
  bool HasVad() const;

  // ---- used by the shared OnlineVadModel (VAD thread) ----
  // Raw samples (VAD sample rate) not yet scored.
  int32_t VadPendingSize() const;
  // Copies the first `win` pending samples; false if fewer are buffered.
  bool VadPeekWindow(int32_t win, std::vector<float> *out) const;
  // Drops `k` scored samples from the front.
  void VadDropSamples(int32_t k);
  // Records one window's decision. On a confirmed speech -> silence flip
  // this arms the endpoint and the end-of-utterance flush.
  void VadPushPrediction(bool is_speech);
  // Model state kept per stream (LSTM state and state-machine counters).
  std::vector<Ort::Value> &GetVadStates();
  void SetVadStates(std::vector<Ort::Value> states);
  bool &IsVadTriggered();
  int32_t &GetVadCurrentSample();
  int32_t &GetVadTempStart();
  int32_t &GetVadTempEnd();

  // Total ASR-feature-frame-equivalent audio the VAD buffer has actually
  // scored so far, in the same absolute frame space as
  // GetNumFramesSinceStart() + GetNumProcessedFrames() (never reset by
  // Reset()/ResetVad()). 0 if no VAD is configured.
  int64_t GetVadFramesCoveredAbsolute() const;

  // True + the absolute frame (same space as above) where the VAD
  // buffer's current confirmed-silence run began, if the VAD is
  // presently silent and a silence run has actually started.
  bool GetVadConfirmedSilenceStartAbsolute(int64_t *out_frame) const;

  // True if the VAD buffer currently considers this stream to contain
  // speech (sticky through its own min-silence-duration hangover).
  bool IsVadSpeechDetected() const;

  // True once the VAD confirmed the end of speech AND the end-of-utterance
  // flush (if any) has been decoded: the endpoint can be reported.
  bool IsVadEndpoint() const;

  // True from when the VAD arms the endpoint until ResetVad().
  bool IsVadEndpointArmed() const;

  // True while the end-of-utterance flush chunk is still to be decoded.
  bool IsVadFlushPending() const;

  // ASR-relative frame (same space as GetNumProcessedFrames()) the flush
  // must decode up to.
  int32_t GetVadEndpointFrame() const;

  void ClearVadFlushPending();

  // Clears the per-segment VAD bookkeeping (seen speech, endpoint, flush).
  // Does not drop the VAD buffer's own continuous state.
  void ResetVad();

  // True once InputFinished() has been called on this stream.
  bool IsInputFinished() const;

  // Which endpoint rule last fired for this stream's current segment: 0
  // none, 1 blank-based (rule1/2/3, undistinguished), 4 VAD-based. Reset
  // to 0 by OnlineRecognizer::Reset().
  int32_t &GetEndpointRule();
  float &GetEndpointDelay();
  float &GetSpeechEndTime();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_STREAM_H_
