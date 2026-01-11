// sherpa-onnx/csrc/axera/online-transducer-greedy-search-decoder-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/online-transducer-greedy-search-decoder-axera.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

OnlineTransducerDecoderResultAxera
OnlineTransducerGreedySearchDecoderAxera::GetEmptyResult() const {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0

  OnlineTransducerDecoderResultAxera r;
  r.tokens.resize(context_size, -1);
  r.tokens.back() = blank_id;

  return r;
}

void OnlineTransducerGreedySearchDecoderAxera::StripLeadingBlanks(
    OnlineTransducerDecoderResultAxera *r) const {
  int32_t context_size = model_->ContextSize();
  auto start = r->tokens.begin() + context_size;
  auto end = r->tokens.end();
  r->tokens = std::vector<int64_t>(start, end);
}

void OnlineTransducerGreedySearchDecoderAxera::Decode(
    std::vector<float> encoder_out,
    OnlineTransducerDecoderResultAxera *result) const {
  auto &r = result[0];

  auto shape = model_->GetEncoderOutShape();
  if (shape.size() < 3) {
    SHERPA_ONNX_LOGE("Encoder output rank is too small. Got rank = %zu",
                     shape.size());
    SHERPA_ONNX_EXIT(-1);
  }

  int32_t num_frames = shape[1];
  int32_t encoder_out_dim = shape[2];

  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();

  std::vector<int32_t> decoder_input;
  std::vector<float> decoder_out;

  auto FillDecoderInput = [&](std::vector<int32_t> *out) {
    out->clear();
    out->reserve(context_size);
    auto start = r.tokens.begin() + (r.tokens.size() - context_size);
    for (auto it = start; it != r.tokens.end(); ++it) {
      out->push_back(static_cast<int32_t>(*it));
    }
  };

  if (r.previous_decoder_out.empty()) {
    FillDecoderInput(&decoder_input);
    decoder_out = model_->RunDecoder(std::move(decoder_input));
  } else {
    decoder_out = std::move(r.previous_decoder_out);
  }

  const float *p_encoder_out = encoder_out.data();
  for (int32_t t = 0; t != num_frames; ++t) {
    auto logit = model_->RunJoiner(p_encoder_out, decoder_out.data());
    p_encoder_out += encoder_out_dim;

    if (blank_penalty_ > 0.0f) {
      logit[0] -= blank_penalty_;
    }

    auto y = static_cast<int32_t>(std::distance(
        logit.data(),
        std::max_element(logit.data(), logit.data() + vocab_size)));

    bool emitted = false;
    // blank id is hardcoded to 0
    // also, it treats unk as blank
    if (y != 0 && y != unk_id_) {
      emitted = true;
      r.tokens.push_back(y);
      r.timestamps.push_back(t + r.frame_offset);
      r.num_trailing_blanks = 0;
    } else {
      ++r.num_trailing_blanks;
    }

    if (emitted) {
      FillDecoderInput(&decoder_input);
      decoder_out = model_->RunDecoder(std::move(decoder_input));
    }
  }

  r.frame_offset += num_frames;
  r.previous_decoder_out = std::move(decoder_out);
}

}  // namespace sherpa_onnx