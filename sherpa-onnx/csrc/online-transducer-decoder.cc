// sherpa-onnx/csrc/online-transducer-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-transducer-decoder.h"

#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

OnlineTransducerDecoderResult::OnlineTransducerDecoderResult(
    const OnlineTransducerDecoderResult &other)
    : OnlineTransducerDecoderResult() {
  *this = other;
}

OnlineTransducerDecoderResult &OnlineTransducerDecoderResult::operator=(
    const OnlineTransducerDecoderResult &other) {
  if (this == &other) {
    return *this;
  }

  tokens = other.tokens;
  num_trailing_blanks = other.num_trailing_blanks;

  Ort::AllocatorWithDefaultOptions allocator;
  if (other.decoder_out) {
    decoder_out = Clone(allocator, &other.decoder_out);
  }

  hyps = other.hyps;

  return *this;
}

OnlineTransducerDecoderResult::OnlineTransducerDecoderResult(
    OnlineTransducerDecoderResult &&other)
    : OnlineTransducerDecoderResult() {
  *this = std::move(other);
}

OnlineTransducerDecoderResult &OnlineTransducerDecoderResult::operator=(
    OnlineTransducerDecoderResult &&other) {
  if (this == &other) {
    return *this;
  }

  tokens = std::move(other.tokens);
  num_trailing_blanks = other.num_trailing_blanks;
  decoder_out = std::move(other.decoder_out);
  hyps = std::move(other.hyps);

  return *this;
}

}  // namespace sherpa_onnx
