// sherpa-onnx/csrc/offline-tts-pocket-zh-en-model.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Please refer to
// https://modelscope.cn/models/dengcunqin/pocket-tts-zh-en
// for the model files.

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_MODEL_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-pocket-zh-en-model-meta-data.h"

namespace sherpa_onnx {

class OfflineTtsPocketZhEnModel {
 public:
  explicit OfflineTtsPocketZhEnModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsPocketZhEnModel(Manager *mgr, const OfflineTtsModelConfig &config);

  ~OfflineTtsPocketZhEnModel();

  // Get model parameters read from ONNX shapes
  const OfflineTtsPocketZhEnModelMetaData &GetMetaData() const;

  // Encode reference audio to voice embedding
  // Input: audio [1, 1, samples]
  // Output: cond [1, frames, model_dim]
  Ort::Value RunEncoder(Ort::Value audio) const;

  // Run one streaming step
  // Input: 12 tensors (tokens, latent, is_bos, cond, gates, noise,
  //        flow_kv, flow_offset, mimi_kv, mimi_offset, mimi_conv, decode_steps)
  // Output: 7 tensors (audio, next_latent, eos_logit,
  //         flow_kv_new, mimi_kv_new, mimi_offset_out, mimi_conv_out)
  std::vector<Ort::Value> RunStep(std::vector<Ort::Value> inputs) const;

  // ===== Helper functions to create initial values =====

  // Create zeros [1, 1, latent_dim]
  Ort::Value GetZeroLatent() const;

  // Create ones [1, 1, 1] (for is_bos at first step)
  Ort::Value GetBosFlag() const;

  // Create zeros [1, 1, 1] (for is_bos after first step)
  Ort::Value GetNonBosFlag() const;

  // gates = [1, 0, 0] for text prefill
  Ort::Value GetTextGates() const;

  // gates = [0, 1, 0] for latent generation
  Ort::Value GetLatentGates() const;

  // gates = [0, 0, 1] for voice conditioning
  Ort::Value GetCondGates() const;

  // Create zeros [1, latent_dim] for noise
  Ort::Value GetZeroNoise() const;

  // Create empty flow_kv [0, flow_layers, 2, 1, flow_heads, flow_head_dim]
  Ort::Value GetEmptyFlowKv() const;

  // Create zero flow_offset (int64 scalar)
  Ort::Value GetZeroFlowOffset() const;

  // Create zero mimi_kv [mimi_kv_len, mimi_layers, 2, 1, mimi_heads,
  // mimi_head_dim]
  Ort::Value GetZeroMimiKv() const;

  // Create zero mimi_offset (int64 scalar)
  Ort::Value GetZeroMimiOffset() const;

  // Create zero mimi_conv [conv_state_size]
  Ort::Value GetZeroMimiConv() const;

  // Create decode_steps = 1.0 (float scalar)
  Ort::Value GetDecodeSteps() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_MODEL_H_
