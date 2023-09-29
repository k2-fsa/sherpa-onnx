// sherpa-onnx/csrc/offline-ctc-fst-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ctc-fst-decoder.h"

#include "fst/fstlib.h"
#include "kaldi_native_io/csrc/kaldi-io.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

// this function is copied from kaldi
fst::Fst<fst::StdArc> *ReadGraph(const std::string &filename) {
  // read decoding network FST
  kaldiio::Input ki(filename);  // use ki.Stream() instead of is.
  if (!ki.Stream().good()) {
    SHERPA_ONNX_LOGE("Could not open decoding-graph FST %s", filename.c_str());
  }

  fst::FstHeader hdr;
  if (!hdr.Read(ki.Stream(), "<unknown>")) {
    SHERPA_ONNX_LOGE("Reading FST: error reading FST header.");
  }

  if (hdr.ArcType() != fst::StdArc::Type()) {
    SHERPA_ONNX_LOGE("FST with arc type %s not supported",
                     hdr.ArcType().c_str());
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst<fst::StdArc> *decode_fst = NULL;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst<fst::StdArc>::Read(ki.Stream(), ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(ki.Stream(), ropts);
  } else {
    SHERPA_ONNX_LOGE("Reading FST: unsupported FST type: %s",
                     hdr.FstType().c_str());
  }

  if (decode_fst == nullptr) {  // fst code will warn.
    SHERPA_ONNX_LOGE("Error reading FST (after reading header).");
    return nullptr;
  } else {
    return decode_fst;
  }
}

OfflineCtcFstDecoder::OfflineCtcFstDecoder(
    const OfflineCtcFstDecoderConfig &config)
    : config_(config), fst_(ReadGraph(config_.graph)) {}

std::vector<OfflineCtcDecoderResult> OfflineCtcFstDecoder::Decode(
    Ort::Value log_probs, Ort::Value log_probs_length) {
  return {};
}

}  // namespace sherpa_onnx
