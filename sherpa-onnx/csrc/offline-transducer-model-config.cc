// sherpa-onnx/csrc/offline-transducer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/offline-transducer-model-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTransducerModelConfig::Register(ParseOptions *po) {
  po->Register("encoder", &encoder_filename, "Path to encoder.onnx");
  po->Register("decoder", &decoder_filename, "Path to decoder.onnx");
  po->Register("joiner", &joiner_filename, "Path to joiner.onnx");
  po->Register("ctc", &ctc, "Path to ctc.onnx");
  po->Register("frame_reducer", &frame_reducer, "Path to frame_reducer.onnx");
  po->Register("encoder_proj", &encoder_proj, "Path to encoder_proj.onnx");
  po->Register("decoder_proj", &decoder_proj, "Path to decoder_proj.onnx");
}

bool OfflineTransducerModelConfig::Validate() const {
  if (!FileExists(encoder_filename)) {
    SHERPA_ONNX_LOGE("transducer encoder: %s does not exist",
                     encoder_filename.c_str());
    return false;
  }

  if (!FileExists(decoder_filename)) {
    SHERPA_ONNX_LOGE("transducer decoder: %s does not exist",
                     decoder_filename.c_str());
    return false;
  }

  if (!FileExists(joiner_filename)) {
    SHERPA_ONNX_LOGE("transducer joiner: %s does not exist",
                     joiner_filename.c_str());
    return false;
  }

  if (!ctc.empty())
  {
    if (!FileExists(ctc)) {
      SHERPA_ONNX_LOGE("ctc: %s does not exist", ctc.c_str());
      return false;
    }

    if (!FileExists(frame_reducer)) {
      SHERPA_ONNX_LOGE("frame_reducer: %s does not exist", frame_reducer.c_str());
      return false;
    }
  }

  if (!encoder_proj.empty())
  {
    if (!FileExists(encoder_proj)) {
      SHERPA_ONNX_LOGE("encoder_proj: %s does not exist", encoder_proj.c_str());
      return false;
    }

    if (!FileExists(decoder_proj)) {
      SHERPA_ONNX_LOGE("decoder_proj: %s does not exist", decoder_proj.c_str());
      return false;
    }    
  }

  return true;
}

std::string OfflineTransducerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTransducerModelConfig(";
  os << "encoder_filename=\"" << encoder_filename << "\", ";
  os << "decoder_filename=\"" << decoder_filename << "\", ";
  os << "joiner_filename=\"" << joiner_filename << "\", ";
  os << "ctc=\"" << ctc << "\", ";
  os << "frame_reducer=\"" << frame_reducer << "\", ";
  os << "encoder_proj=\"" << encoder_proj << "\", ";
  os << "decoder_proj=\"" << decoder_proj << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
