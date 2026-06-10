// sherpa-onnx/csrc/offline-catt-model.h
//
// Copyright (c)  2026  Matias Lin
#ifndef SHERPA_ONNX_CSRC_OFFLINE_CATT_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CATT_MODEL_H_

#include <memory>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-diacritization-model-config.h"

namespace sherpa_onnx {

/**
 * Encoder-Only variant of the CATT (Context-Aware Transformer for Tashkeel)
 * Arabic diacritization model. It owns two Ort sessions: an encoder and a
 * non-autoregressive classifier head ("decoder").
 * Reference: https://github.com/abjadai/catt/blob/main/catt_models_onnx.py
 */
class OfflineCATTModel {
 public:
  explicit OfflineCATTModel(const OfflineDiacritizationModelConfig &config);

  template <typename Manager>
  OfflineCATTModel(Manager *mgr,
                   const OfflineDiacritizationModelConfig &config);
  ~OfflineCATTModel();

  /**
   * Run the encoder model
   * @param src       A tensor of shape (N, T) of type int64
   * @param src_mask  A tensor of shape (N, 1, T, T) of type bool
   * @return enc_src  A tensor of shape (N, T, D) of type float
   */
  Ort::Value RunEncoder(Ort::Value src, Ort::Value src_mask) const;
  /**
   * Run the decoder model
   * @param enc_src  A tensor of shape (N, T, D) of type float
   * @return logits  A tensor of shape (N, T, V) of type float
   */
  Ort::Value RunDecoder(Ort::Value enc_src) const;

  /**
   * Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CATT_MODEL_H_
