// sherpa-onnx/csrc/online-transducer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_H_

#include <memory>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-model-config.h"

namespace sherpa_onnx {

class OnlineTransducerDecoderResult;

class OnlineTransducerModel {
 public:
  virtual ~OnlineTransducerModel() = default;

  static std::unique_ptr<OnlineTransducerModel> Create(
      const OnlineTransducerModelConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<OnlineTransducerModel> Create(
      AAssetManager *mgr, const OnlineTransducerModelConfig &config);
#endif

  /** Stack a list of individual states into a batch.
   *
   * It is the inverse operation of `UnStackStates`.
   *
   * @param states states[i] contains the state for the i-th utterance.
   * @return Return a single value representing the batched state.
   */
  virtual std::vector<Ort::Value> StackStates(
      const std::vector<std::vector<Ort::Value>> &states) const = 0;

  /** Unstack a batch state into a list of individual states.
   *
   * It is the inverse operation of `StackStates`.
   *
   * @param states A batched state.
   * @return ans[i] contains the state for the i-th utterance.
   */
  virtual std::vector<std::vector<Ort::Value>> UnStackStates(
      const std::vector<Ort::Value> &states) const = 0;

  /** Get the initial encoder states.
   *
   * @return Return the initial encoder state.
   */
  virtual std::vector<Ort::Value> GetEncoderInitStates() = 0;

  /** Run the encoder.
   *
   * @param features  A tensor of shape (N, T, C). It is changed in-place.
   * @param states  Encoder state of the previous chunk. It is changed in-place.
   *
   * @return Return a tuple containing:
   *           - encoder_out, a tensor of shape (N, T', encoder_out_dim)
   *           - next_states  Encoder state for the next chunk.
   */
  virtual std::pair<Ort::Value, std::vector<Ort::Value>> RunEncoder(
      Ort::Value features,
      std::vector<Ort::Value> states) = 0;  // NOLINT

  /** Run the decoder network.
   *
   * Caution: We assume there are no recurrent connections in the decoder and
   *          the decoder is stateless. See
   * https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless2/decoder.py
   *          for an example
   *
   * @param decoder_input It is usually of shape (N, context_size)
   * @return Return a tensor of shape (N, decoder_dim).
   */
  virtual Ort::Value RunDecoder(Ort::Value decoder_input) = 0;

  /** Run the joint network.
   *
   * @param encoder_out Output of the encoder network. A tensor of shape
   *                    (N, joiner_dim).
   * @param decoder_out Output of the decoder network. A tensor of shape
   *                    (N, joiner_dim).
   * @return Return a tensor of shape (N, vocab_size). In icefall, the last
   *         last layer of the joint network is `nn.Linear`,
   *         not `nn.LogSoftmax`.
   */
  virtual Ort::Value RunJoiner(Ort::Value encoder_out,
                               Ort::Value decoder_out) = 0;

  /** If we are using a stateless decoder and if it contains a
   *  Conv1D, this function returns the kernel size of the convolution layer.
   */
  virtual int32_t ContextSize() const = 0;

  /** We send this number of feature frames to the encoder at a time. */
  virtual int32_t ChunkSize() const = 0;

  /** Number of input frames to discard after each call to RunEncoder.
   *
   * For instance, if we have 30 frames, chunk_size=8, chunk_shift=6.
   *
   * In the first call of RunEncoder, we use frames 0~7 since chunk_size is 8.
   * Then we discard frame 0~5 since chunk_shift is 6.
   * In the second call of RunEncoder, we use frames 6~13; and then we discard
   * frames 6~11.
   * In the third call of RunEncoder, we use frames 12~19; and then we discard
   * frames 12~16.
   *
   * Note: ChunkSize() - ChunkShift() == right context size
   */
  virtual int32_t ChunkShift() const = 0;

  virtual int32_t VocabSize() const = 0;

  virtual int32_t SubsamplingFactor() const { return 4; }

  virtual OrtAllocator *Allocator() = 0;

  Ort::Value BuildDecoderInput(
      const std::vector<OnlineTransducerDecoderResult> &results);

  Ort::Value BuildDecoderInput(const std::vector<Hypothesis> &hyps);
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_H_
