// sherpa-onnx/csrc/offline-tts-pocket-model.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_MODEL_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

struct PocketLmMainState {
  std::vector<Ort::Value> values;
};

struct PocketMimiDecoderState {
  std::vector<Ort::Value> values;
};

// Please refer to
// https://huggingface.co/KevinAHM/pocket-tts-onnx/blob/main/pocket_tts_onnx.py
class OfflineTtsPocketModel {
 public:
  explicit OfflineTtsPocketModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsPocketModel(Manager *mgr, const OfflineTtsModelConfig &config);

  ~OfflineTtsPocketModel();

  PocketLmMainState GetLmMainInitState() const;
  PocketMimiDecoderState GetMimiDecoderInitState() const;

  /**
   * @param audio should be of 24000Hz. Its shape is (1, 1, num_samples)
   * @returns a float32 tensor of shape (1, num_frames, 1024)
   */
  Ort::Value RunMimiEncoder(Ort::Value audio) const;

  /**
   * @param text_tokens (1, num_tokens) of shape int64
   * @return float32 tensor of shape (1, num_tokens, 1024)
   */
  Ort::Value RunTextConditioner(Ort::Value text_tokens) const;

  Ort::Value RunLmFlow(Ort::Value c, Ort::Value s, Ort::Value t,
                       Ort::Value x) const;

  std::tuple<Ort::Value, Ort::Value, PocketLmMainState> RunLmMain(
      Ort::Value seq, Ort::Value embeddings, PocketLmMainState state) const;

  std::pair<Ort::Value, PocketMimiDecoderState> RunMimiDecoder(
      Ort::Value latent, PocketMimiDecoderState state) const;

  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_MODEL_H_
