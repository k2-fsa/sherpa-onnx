// sherpa-onnx/csrc/offline-tts-pocket-zh-en-model-meta-data.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_MODEL_META_DATA_H_

#include <cstdint>

namespace sherpa_onnx {

struct OfflineTtsPocketZhEnModelMetaData {
  // From step_encoder.onnx output[0]: cond [1, frames, model_dim]
  int32_t model_dim = 0;

  // From step_model.onnx inputs:
  int32_t latent_dim = 0;       // input[5] noise [1, 32]
  int32_t flow_layers = 0;      // input[6] flow_kv [past, 6, 2, 1, 16, 64]
  int32_t flow_heads = 0;
  int32_t flow_head_dim = 0;
  int32_t mimi_layers = 0;      // input[8] mimi_kv [266, 2, 2, 1, 8, 64]
  int32_t mimi_heads = 0;
  int32_t mimi_head_dim = 0;
  int32_t mimi_kv_len = 0;
  int32_t conv_state_size = 0;  // input[10] mimi_conv [14720]
  int32_t frame_size = 0;       // output[0] audio [1, 1, 1920]

  // Fixed
  int32_t sample_rate = 24000;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_MODEL_META_DATA_H_
