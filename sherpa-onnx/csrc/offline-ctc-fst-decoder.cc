// sherpa-onnx/csrc/offline-ctc-fst-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ctc-fst-decoder.h"

#include <string>
#include <utility>

#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/decodable-ctc.h"
#include "kaldi-decoder/csrc/eigen.h"
#include "kaldi-decoder/csrc/faster-decoder.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

// This function is copied from kaldi.
//
// @param filename Path to a StdVectorFst or StdConstFst graph
// @return The caller should free the returned pointer using `delete` to
//         avoid memory leak.
static fst::Fst<fst::StdArc> *ReadGraph(const std::string &filename) {
  // read decoding network FST
  std::ifstream is(filename, std::ios::binary);
  if (!is.good()) {
    SHERPA_ONNX_LOGE("Could not open decoding-graph FST %s", filename.c_str());
  }

  fst::FstHeader hdr;
  if (!hdr.Read(is, "<unknown>")) {
    SHERPA_ONNX_LOGE("Reading FST: error reading FST header.");
  }

  if (hdr.ArcType() != fst::StdArc::Type()) {
    SHERPA_ONNX_LOGE("FST with arc type %s not supported",
                     hdr.ArcType().c_str());
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst<fst::StdArc> *decode_fst = nullptr;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst<fst::StdArc>::Read(is, ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(is, ropts);
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

/**
 * @param decoder
 * @param p Pointer to a 2-d array of shape (num_frames, vocab_size)
 * @param num_frames Number of rows in the 2-d array.
 * @param vocab_size Number of columns in the 2-d array.
 * @return Return the decoded result.
 */
static OfflineCtcDecoderResult DecodeOne(kaldi_decoder::FasterDecoder *decoder,
                                         const float *p, int32_t num_frames,
                                         int32_t vocab_size) {
  OfflineCtcDecoderResult r;
  kaldi_decoder::DecodableCtc decodable(p, num_frames, vocab_size);

  decoder->Decode(&decodable);

  if (!decoder->ReachedFinal()) {
    SHERPA_ONNX_LOGE("Not reached final!");
    return r;
  }

  fst::VectorFst<fst::LatticeArc> decoded;  // linear FST.
  decoder->GetBestPath(&decoded);

  if (decoded.NumStates() == 0) {
    SHERPA_ONNX_LOGE("Empty best path!");
    return r;
  }

  auto cur_state = decoded.Start();

  int32_t blank_id = 0;

  for (int32_t t = 0, prev = -1; decoded.NumArcs(cur_state) == 1; ++t) {
    fst::ArcIterator<fst::Fst<fst::LatticeArc>> iter(decoded, cur_state);
    const auto &arc = iter.Value();

    cur_state = arc.nextstate;

    if (arc.ilabel == prev) {
      continue;
    }

    // 0 is epsilon here
    if (arc.ilabel == 0 || arc.ilabel == blank_id + 1) {
      prev = arc.ilabel;
      continue;
    }

    // -1 here since the input labels are incremented during graph
    // construction
    r.tokens.push_back(arc.ilabel - 1);

    r.timestamps.push_back(t);
    prev = arc.ilabel;
  }

  return r;
}

OfflineCtcFstDecoder::OfflineCtcFstDecoder(
    const OfflineCtcFstDecoderConfig &config)
    : config_(config), fst_(ReadGraph(config_.graph)) {}

std::vector<OfflineCtcDecoderResult> OfflineCtcFstDecoder::Decode(
    Ort::Value log_probs, Ort::Value log_probs_length) {
  std::vector<int64_t> shape = log_probs.GetTensorTypeAndShapeInfo().GetShape();

  assert(static_cast<int32_t>(shape.size()) == 3);
  int32_t batch_size = shape[0];
  int32_t T = shape[1];
  int32_t vocab_size = shape[2];

  std::vector<int64_t> length_shape =
      log_probs_length.GetTensorTypeAndShapeInfo().GetShape();
  assert(static_cast<int32_t>(length_shape.size()) == 1);

  assert(shape[0] == length_shape[0]);

  kaldi_decoder::FasterDecoderOptions opts;
  opts.max_active = config_.max_active;
  kaldi_decoder::FasterDecoder faster_decoder(*fst_, opts);

  const float *start = log_probs.GetTensorData<float>();

  std::vector<OfflineCtcDecoderResult> ans;
  ans.reserve(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    const float *p = start + i * T * vocab_size;
    int32_t num_frames = log_probs_length.GetTensorData<int64_t>()[i];
    auto r = DecodeOne(&faster_decoder, p, num_frames, vocab_size);
    ans.push_back(std::move(r));
  }

  return ans;
}

}  // namespace sherpa_onnx
