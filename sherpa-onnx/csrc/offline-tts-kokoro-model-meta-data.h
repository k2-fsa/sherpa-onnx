// sherpa-onnx/csrc/offline-tts-kokoro-model-metadata.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_MODEL_META_DATA_H_

#include <cstdint>
#include <string>

namespace sherpa_onnx {

// please refer to
// https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/kokoro/add-meta-data.py
struct OfflineTtsKokoroModelMetaData {
  int32_t sample_rate = 0;
  int32_t num_speakers = 0;
  int32_t version = 1;
  int32_t has_espeak = 1;
  int32_t max_token_len = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_MODEL_META_DATA_H_
