// sherpa-onnx/csrc/offline-moonshine-model-v2.h
//
// Copyright (c)  2024-2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_MODEL_V2_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_MODEL_V2_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

// please see
// https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/moonshine/merged/test.py
class OfflineMoonshineModelV2 {
 public:
  explicit OfflineMoonshineModelV2(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineMoonshineModelV2(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineMoonshineModelV2();

  /** Run the encoder model.
   *
   * @param audio A float32 tensor of shape (batch_size, num_samples)
   *
   * @return Return a float32 tensor of shape (batch_size, T, dim) that
   *         can be used as the input of ForwardDecoder()
   *
   * Note it currently supports only batch size 1.
   */
  Ort::Value ForwardEncoder(Ort::Value audio) const;

  /** Run the merged decoder.
   *
   * @param token A int64 tensor of shape (batch_size, num_tokens)
   * @param encoder_out A float32 tensor of shape (batch_size, T, dim)
   * @param states Model States
   *
   * @returns Return a pair:
   *
   *          - logits, a float32 tensor of shape (batch_size, 1, dim)
   *          - states, a list of states
   *
   * Note it supports only batch_size 1.
   */
  std::pair<Ort::Value, std::vector<Ort::Value>> ForwardDecoder(
      Ort::Value token, Ort::Value encoder_out,
      std::vector<Ort::Value> states) const;

  std::vector<Ort::Value> GetDecoderInitStates() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_MODEL_V2_H_
