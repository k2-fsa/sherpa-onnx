// sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/packed-sequence.h"
#include "sherpa-onnx/csrc/slice.h"

namespace sherpa_onnx {

/**
 * @brief Greedy-decodes transducer model encoder outputs into token sequences for each batch item.
 *
 * Decodes the packed, padded encoder outputs using the model's decoder and joiner
 * in a greedy left-to-right manner. For each emitted non-blank and non-unknown
 * token the corresponding timestamp and log-probability (log-softmax) are
 * recorded. The initial decoder context tokens are removed from each result and
 * the returned vector is reordered back to the original batch order.
 *
 * @param encoder_out Encoder output tensor (padded) for the batch.
 * @param encoder_out_length Lengths for each sequence in the batch.
 * @param ss Optional array of OfflineStream pointers associated with the batch (may be nullptr).
 * @param n Optional integer parameter (reserved for compatibility; not required for typical use).
 * @return std::vector<OfflineTransducerDecoderResult> Decoding results for each batch item in the original order.
 *
 * @note The implementation treats token ID 0 as the blank token and does not emit blank or `unk_id_` tokens.
 */
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
    float *p_logit = logit.GetTensorMutableData<float>();
    bool emitted = false;
    for (int32_t i = 0; i != n; ++i) {
      if (blank_penalty_ > 0.0) {
        p_logit[0] -= blank_penalty_;  // assuming blank id is 0
      }

      // Compute log softmax to get log probabilities
      float max_logit = *std::max_element(p_logit, p_logit + vocab_size);
      float sum_exp = 0.0f;
      for (int32_t k = 0; k != vocab_size; ++k) {
        sum_exp += std::exp(p_logit[k] - max_logit);
      }
      float log_sum_exp = max_logit + std::log(sum_exp);

      auto y = static_cast<int32_t>(std::distance(
          static_cast<const float *>(p_logit),
          std::max_element(static_cast<const float *>(p_logit),
                           static_cast<const float *>(p_logit) + vocab_size)));

      // Compute log probability for the selected token
      float log_prob = p_logit[y] - log_sum_exp;

      p_logit += vocab_size;
      // blank id is hardcoded to 0
      // also, it treats unk as blank
      if (y != 0 && y != unk_id_) {
        ans[i].tokens.push_back(y);
        ans[i].timestamps.push_back(t);
        ans[i].ys_probs.push_back(log_prob);
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