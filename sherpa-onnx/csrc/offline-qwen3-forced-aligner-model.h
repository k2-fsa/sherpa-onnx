// sherpa-onnx/csrc/offline-qwen3-forced-aligner-model.h
//
// Copyright (c)  2026  losewayy

#ifndef SHERPA_ONNX_CSRC_OFFLINE_QWEN3_FORCED_ALIGNER_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_QWEN3_FORCED_ALIGNER_MODEL_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

// This class implements the Qwen3-ForcedAligner model
// (Qwen/Qwen3-ForcedAligner-0.6B). Given log-mel features and a token
// sequence containing <timestamp> placeholder slots, it predicts, for each
// slot, a timestamp index in units of 80 ms (classify_num = 5000, i.e. up to
// 400 seconds of audio).
//
// The model is exported as three ONNX graphs, sharing the audio tower layout
// with Qwen3-ASR:
//   - conv_frontend.onnx: [B, T, 128] -> [B, A, 1024]
//   - encoder.onnx: ([B, A, 1024], [B, A] bool) -> [B, A, 1024]
//   - decoder.onnx: (input_ids [B, S], audio_features [B, A, 1024],
//                    attention_mask [B, S]) -> logits [B, S, 5000]
// The decoder performs a single non-autoregressive forward pass.
class OfflineQwen3ForcedAlignerModel {
 public:
  explicit OfflineQwen3ForcedAlignerModel(const OfflineModelConfig &config);

#if __ANDROID_API__ >= 9
  OfflineQwen3ForcedAlignerModel(AAssetManager *mgr,
                                 const OfflineModelConfig &config);
#endif

#if __OHOS__
  OfflineQwen3ForcedAlignerModel(NativeResourceManager *mgr,
                                 const OfflineModelConfig &config);
#endif

  ~OfflineQwen3ForcedAlignerModel();

  // input_features: [B, T, 128] log-mel, returns [B, A, 1024]
  Ort::Value ForwardConvFrontend(Ort::Value input_features) const;

  // conv_output: [B, A, 1024], feature_attention_mask: [B, A] bool
  // returns audio features [B, A, 1024]
  Ort::Value ForwardEncoder(Ort::Value conv_output,
                            Ort::Value feature_attention_mask) const;

  // input_ids: [B, S] int64, audio_features: [B, A, 1024],
  // attention_mask: [B, S] int64. Returns logits [B, S, 5000] float32.
  Ort::Value ForwardDecoder(Ort::Value input_ids, Ort::Value audio_features,
                            Ort::Value attention_mask) const;

  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_QWEN3_FORCED_ALIGNER_MODEL_H_
