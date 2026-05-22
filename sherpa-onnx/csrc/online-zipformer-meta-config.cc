// sherpa-onnx/csrc/online-zipformer-meta-config.cc

#include "sherpa-onnx/csrc/online-zipformer-meta-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/vec-to-string.h"

namespace sherpa_onnx {

void OnlineZipformerMetaConfig::Register(ParseOptions *po) {
  po->Register("zipformer-encoder-dims", &encoder_dims,
               "Zipformer encoder dims, e.g., 384,384,384,384");
  po->Register("zipformer-attention-dims", &attention_dims,
               "Zipformer attention dims, e.g., 192,192,192,192");
  po->Register("zipformer-num-encoder-layers", &num_encoder_layers,
               "Zipformer num encoder layers per stage, e.g., 2,2,2,2");
  po->Register("zipformer-cnn-module-kernels", &cnn_module_kernels,
               "Zipformer cnn module kernels per stage, e.g., 31,31,31,31");
  po->Register("zipformer-left-context-len", &left_context_len,
               "Zipformer left context length per stage, e.g., 32,32,32,32");

  po->Register("zipformer-T", &T, "Zipformer chunk size (frames)");
  po->Register("zipformer-decode-chunk-len", &decode_chunk_len,
               "Zipformer chunk shift (frames)");
  po->Register("zipformer-context-size", &context_size,
               "Zipformer transducer decoder context size (usually 2)");
}

bool OnlineZipformerMetaConfig::Validate() const {
  if (encoder_dims.empty() || attention_dims.empty() ||
      num_encoder_layers.empty() || cnn_module_kernels.empty() ||
      left_context_len.empty()) {
    return false;
  }
  if (T <= 0 || decode_chunk_len <= 0 || context_size <= 0) {
    return false;
  }

  size_t n = encoder_dims.size();
  if (attention_dims.size() != n || num_encoder_layers.size() != n ||
      cnn_module_kernels.size() != n || left_context_len.size() != n) {
    return false;
  }

  return true;
}

std::string OnlineZipformerMetaConfig::ToString() const {
  std::ostringstream os;
  os << "OnlineZipformerMetaConfig(";
  os << "encoder_dims=" << VecToString(encoder_dims) << ", ";
  os << "attention_dims=" << VecToString(attention_dims) << ", ";
  os << "num_encoder_layers=" << VecToString(num_encoder_layers) << ", ";
  os << "cnn_module_kernels=" << VecToString(cnn_module_kernels) << ", ";
  os << "left_context_len=" << VecToString(left_context_len) << ", ";
  os << "T=" << T << ", ";
  os << "decode_chunk_len=" << decode_chunk_len << ", ";
  os << "context_size=" << context_size << ")";
  return os.str();
}

}  // namespace sherpa_onnx