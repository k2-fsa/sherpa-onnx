// sherpa-onnx/csrc/offline-tts-vits-model-metadata.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_METADATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_METADATA_H_

#include <cstdint>
#include <string>

namespace sherpa_onnx {

// If you are not sure what each field means, please
// have a look of the Python file in the model directory that
// you have downloaded.
struct OfflineTtsVitsModelMetaData {
  int32_t sample_rate = 0;
  int32_t add_blank = 0;
  int32_t num_speakers = 0;

  bool is_piper = false;
  bool is_coqui = false;

  // the following options are for models from coqui-ai/TTS
  int32_t blank_id = 0;
  int32_t bos_id = 0;
  int32_t eos_id = 0;
  int32_t use_eos_bos = 0;
  int32_t pad_id = 0;

  std::string punctuations;
  std::string language;
  std::string voice;
  std::string frontend;  // characters
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_METADATA_H_
