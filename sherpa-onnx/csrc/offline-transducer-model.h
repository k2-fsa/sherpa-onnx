// sherpa-onnx/csrc/offline-transducer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_H_

#include <memory>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

struct OfflineTransducerDecoderResult;

class OfflineTransducerModel {
 public:
  explicit OfflineTransducerModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineTransducerModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineTransducerModel();

  /** Run the encoder.
   *
   * @param features  A tensor of shape (N, T, C). It is changed in-place.
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *                         Its dtype is int64_t.
   *
   * @return Return a pair containing:
   *  - encoder_out: A 3-D tensor of shape (N, T', encoder_dim)
   *  - encoder_out_length: A 1-D tensor of shape (N,) containing number
   *                        of frames in `encoder_out` before padding.
   */
  std::pair<Ort::Value, Ort::Value> RunEncoder(Ort::Value features,
                                               Ort::Value features_length);

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
  Ort::Value RunDecoder(Ort::Value decoder_input);

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
  Ort::Value RunJoiner(Ort::Value encoder_out, Ort::Value decoder_out);

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const;

  /** Return the context_size of the decoder model.
   */
  int32_t ContextSize() const;

  /** Return the subsampling factor of the model.
   */
  int32_t SubsamplingFactor() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

  /** Build decoder_input from the current results.
   *
   * @param results Current decoded results.
   * @param end_index We only use results[0:end_index] to build
   *                  the decoder_input. results[end_index] is not used.
   * @return Return a tensor of shape (results.size(), ContextSize())
   */
  Ort::Value BuildDecoderInput(
      const std::vector<OfflineTransducerDecoderResult> &results,
      int32_t end_index) const;

  Ort::Value BuildDecoderInput(const std::vector<Hypothesis> &results,
                               int32_t end_index) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_H_
