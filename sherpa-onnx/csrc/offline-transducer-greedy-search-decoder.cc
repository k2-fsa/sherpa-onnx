// sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.h"

#include <algorithm>
#include <iterator>
#include <utility>

#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/packed-sequence.h"
#include "sherpa-onnx/csrc/slice.h"

namespace sherpa_onnx {

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerGreedySearchDecoder::Decode(Ort::Value encoder_out,
                                             Ort::Value encoder_out_length,
                                             OfflineStream **ss /*= nullptr*/,
                                             int32_t n /*= 0*/) {
  PackedSequence packed_encoder_out = PackPaddedSequence(
      model_->Allocator(), &encoder_out, &encoder_out_length);

  int32_t batch_size =
      static_cast<int32_t>(packed_encoder_out.sorted_indexes.size());

  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();

  std::vector<OfflineTransducerDecoderResult> ans(batch_size);
  for (auto &r : ans) {
    r.tokens.resize(context_size, -1);
    // 0 is the ID of the blank token
    r.tokens.back() = 0;
  }

  auto decoder_input = model_->BuildDecoderInput(ans, ans.size());
  Ort::Value decoder_out = model_->RunDecoder(std::move(decoder_input));

  int32_t start = 0;
  int32_t t = 0;
  for (auto n : packed_encoder_out.batch_sizes) {
    Ort::Value cur_encoder_out = packed_encoder_out.Get(start, n);
    Ort::Value cur_decoder_out = Slice(model_->Allocator(), &decoder_out, 0, n);
    start += n;
    Ort::Value logit = model_->RunJoiner(std::move(cur_encoder_out),
                                         std::move(cur_decoder_out));
    const float *p_logit = logit.GetTensorData<float>();
    bool emitted = false;
    for (int32_t i = 0; i != n; ++i) {
      auto y = static_cast<int32_t>(std::distance(
          static_cast<const float *>(p_logit),
          std::max_element(static_cast<const float *>(p_logit),
                           static_cast<const float *>(p_logit) + vocab_size)));
      p_logit += vocab_size;
      if (y != 0) {
        ans[i].tokens.push_back(y);
        ans[i].timestamps.push_back(t);
        emitted = true;
      }
    }
    if (emitted) {
      Ort::Value decoder_input = model_->BuildDecoderInput(ans, n);
      decoder_out = model_->RunDecoder(std::move(decoder_input));
    }
    ++t;
  }

  for (auto &r : ans) {
    r.tokens = {r.tokens.begin() + context_size, r.tokens.end()};
  }

  std::vector<OfflineTransducerDecoderResult> unsorted_ans(batch_size);
  for (int32_t i = 0; i != batch_size; ++i) {
    unsorted_ans[packed_encoder_out.sorted_indexes[i]] = std::move(ans[i]);
  }

  return unsorted_ans;
}

}  // namespace sherpa_onnx
