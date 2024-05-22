// sherpa-onnx/csrc/online-transducer-nemo-model.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-model-config.h"

namespace sherpa_onnx {

// see
// https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/hybrid_rnnt_ctc_bpe_models.py#L40
// Its decoder is stateful, not stateless.
class OnlineTransducerNeMoModel {
 public:
  explicit OnlineTransducerNeMoModel(const OnlineModelConfig &config);

#if __ANDROID_API__ >= 9
  OnlineTransducerNeMoModel(AAssetManager *mgr,
                            const OfflineModelConfig &config);
#endif
  
  ~OnlineTransducerNeMoModel();
    // A list of 3 tensors:
  //  - cache_last_channel
  //  - cache_last_time
  //  - cache_last_channel_len
  std::vector<Ort::Value> GetInitStates() const;

  /** Stack a list of individual states into a batch.
   *
   * It is the inverse operation of `UnStackStates`.
   *
   * @param states states[i] contains the state for the i-th utterance.
   * @return Return a single value representing the batched state.
   */
  std::vector<Ort::Value> StackStates(
      const std::vector<std::vector<Ort::Value>> &states) const;

  /** Unstack a batch state into a list of individual states.
   *
   * It is the inverse operation of `StackStates`.
   *
   * @param states A batched state.
   * @return ans[i] contains the state for the i-th utterance.
   */
  std::vector<std::vector<Ort::Value>> UnStackStates(
      const std::vector<Ort::Value> &states) const;

  // /** Get the initial encoder states.
  //  *
  //  * @return Return the initial encoder state.
  //  */
  // std::vector<Ort::Value> GetEncoderInitStates() = 0;

  /** Run the encoder.
   *
   * @param features  A tensor of shape (N, T, C). It is changed in-place.
   * @param states  Encoder state of the previous chunk. It is changed in-place.
   * @param processed_frames  Processed frames before subsampling. It is a 1-D
   * tensor with data type int64_t.
   *
   * @return Return a tuple containing:
   *           - encoder_out, a tensor of shape (N, T', encoder_out_dim)
   *           - next_states  Encoder state for the next chunk.
   */
  std::pair<std::vector<Ort::Value>, std::vector<Ort::Value>> RunEncoder(
      Ort::Value features, std::vector<Ort::Value> states,
      Ort::Value processed_frames) const;  // NOLINT

  /** Run the decoder network.
   *
   * @param targets A int32 tensor of shape (batch_size, 1)
   * @param targets_length A int32 tensor of shape (batch_size,)
   * @param states The states for the decoder model.
   * @return Return a vector:
   *           - ans[0] is the decoder_out (a float tensor)
   *           - ans[1] is the decoder_out_length (a int32 tensor)
   *           - ans[2:] is the states_next
   */
  std::pair<Ort::Value, std::vector<Ort::Value>> RunDecoder(
      Ort::Value targets, Ort::Value targets_length,
      std::vector<Ort::Value> states) const;

  std::vector<Ort::Value> GetDecoderInitStates(int32_t batch_size) const;

  /** Run the joint network.
   *
   * @param encoder_out Output of the encoder network.
   * @param decoder_out Output of the decoder network.
   * @return Return a tensor of shape (N, 1, 1, vocab_size) containing logits.
   */
  virtual Ort::Value RunJoiner( Ort::Value encoder_out, 
                                Ort::Value decoder_out) const;

  // cache_last_time_dim3 in the model meta_data
  int32_t ContextSize() const;

  /** We send this number of feature frames to the encoder at a time. */
  int32_t ChunkSize() const;

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
  int32_t ChunkShift() const;

  /** Return the subsampling factor of the model.
   */
  int32_t SubsamplingFactor() const;

  int32_t VocabSize() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

  // Possible values:
  // - per_feature
  // - all_features (not implemented yet)
  // - fixed_mean (not implemented)
  // - fixed_std (not implemented)
  // - or just leave it to empty
  // See
  // https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L59
  // for details
  std::string FeatureNormalizationMethod() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_MODEL_H_
