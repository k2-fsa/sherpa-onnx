// sherpa-onnx/csrc/offline-transducer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_H_

#include <memory>
#include <utility>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-transducer-model-config.h"

namespace sherpa_onnx {

class OfflineTransducerModel {
 public:
  explicit OfflineTransducerModel(const OfflineTransducerModelConfig &config);
  ~OfflineTransducerModel();

  /** Run the encoder.
   *
   * @param features  A tensor of shape (N, T, C). It is changed in-place.
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
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

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODEL_H_
