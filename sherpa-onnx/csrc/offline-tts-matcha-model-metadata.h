// sherpa-onnx/csrc/offline-tts-matcha-model-metadata.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_METADATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_METADATA_H_

#include <cstdint>
#include <string>

namespace sherpa_onnx {

// If you are not sure what each field means, please
// have a look of the Python file in the model directory that
// you have downloaded.
struct OfflineTtsMatchaModelMetaData {
  int32_t sample_rate = 0;
  int32_t add_blank = 0;
  int32_t num_speakers = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_METADATA_H_
