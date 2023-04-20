// sherpa-onnx/csrc/offline-transducer-modified-beam-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-transducer-modified-beam-search-decoder.h"

#include <deque>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/packed-sequence.h"
#include "sherpa-onnx/csrc/slice.h"

namespace sherpa_onnx {

static std::vector<int32_t> GetHypsRowSplits(
    const std::vector<Hypotheses> &hyps) {
  std::vector<int32_t> row_splits;
  row_splits.reserve(hyps.size() + 1);

  row_splits.push_back(0);
  int32_t s = 0;
  for (const auto &h : hyps) {
    s += h.Size();
    row_splits.push_back(s);
  }

  return row_splits;
}

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerModifiedBeamSearchDecoder::Decode(
    Ort::Value encoder_out, Ort::Value encoder_out_length) {
  PackedSequence packed_encoder_out = PackPaddedSequence(
      model_->Allocator(), &encoder_out, &encoder_out_length);

  int32_t batch_size =
      static_cast<int32_t>(packed_encoder_out.sorted_indexes.size());

  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();

  std::vector<int64_t> blanks(context_size, 0);
  Hypotheses blank_hyp({{blanks, 0}});

  std::deque<Hypotheses> finalized;
  std::vector<Hypotheses> cur(batch_size, blank_hyp);
  std::vector<Hypothesis> prev;

  int32_t start = 0;
  int32_t t = 0;
  for (auto n : packed_encoder_out.batch_sizes) {
    Ort::Value cur_encoder_out = packed_encoder_out.Get(start, n);
    start += n;

    if (n < static_cast<int32_t>(cur.size())) {
      for (int32_t k = static_cast<int32_t>(cur.size()) - 1; k >= n; --k) {
        finalized.push_front(std::move(cur[k]));
      }

      cur.erase(cur.begin() + n, cur.end());
    }  // if (n < static_cast<int32_t>(cur.size()))

    // Due to merging paths with identical token sequences,
    // not all utterances have "max_active_paths" paths.
    auto hyps_row_splits = GetHypsRowSplits(cur);
    int32_t num_hyps = hyps_row_splits.back();

    prev.clear();
    prev.reserve(num_hyps);

    for (auto &hyps : cur) {
      for (auto &h : hyps) {
        prev.push_back(std::move(h.second));
      }
    }
    cur.clear();
    cur.reserve(n);

    auto decoder_input = model_->BuildDecoderInput(prev, num_hyps);
    // decoder_input shape: (num_hyps, context_size)

    auto decoder_out = model_->RunDecoder(std::move(decoder_input));
    // decoder_out is (num_hyps, joiner_dim)

    cur_encoder_out =
        Repeat(model_->Allocator(), &cur_encoder_out, hyps_row_splits);
    // now cur_encoder_out is of shape (num_hyps, joiner_dim)

    Ort::Value logit = model_->RunJoiner(
        std::move(cur_encoder_out), Clone(model_->Allocator(), &decoder_out));

    float *p_logit = logit.GetTensorMutableData<float>();
    LogSoftmax(p_logit, vocab_size, num_hyps);

    // now p_logit contains log_softmax output, we rename it to p_logprob
    // to match what it actually contains
    float *p_logprob = p_logit;

    // add log_prob of each hypothesis to p_logprob before taking top_k
    for (int32_t i = 0; i != num_hyps; ++i) {
      float log_prob = prev[i].log_prob;
      for (int32_t k = 0; k != vocab_size; ++k, ++p_logprob) {
        *p_logprob += log_prob;
      }
    }
    p_logprob = p_logit;  // we changed p_logprob in the above for loop

    // Now compute top_k for each utterance
    for (int32_t i = 0; i != n; ++i) {
      int32_t start = hyps_row_splits[i];
      int32_t end = hyps_row_splits[i + 1];
      auto topk =
          TopkIndex(p_logprob, vocab_size * (end - start), max_active_paths_);

      Hypotheses hyps;
      for (auto k : topk) {
        int32_t hyp_index = k / vocab_size + start;
        int32_t new_token = k % vocab_size;
        Hypothesis new_hyp = prev[hyp_index];

        if (new_token != 0) {
          // blank id is fixed to 0
          new_hyp.ys.push_back(new_token);
          new_hyp.timestamps.push_back(t);
        }

        new_hyp.log_prob = p_logprob[k];
        hyps.Add(std::move(new_hyp));
      }  // for (auto k : topk)
      p_logprob += (end - start) * vocab_size;
      cur.push_back(std::move(hyps));
    }  // for (int32_t i = 0; i != n; ++i)

    ++t;
  }  // for (auto n : packed_encoder_out.batch_sizes)

  for (auto &h : finalized) {
    cur.push_back(std::move(h));
  }

  if (lm_) {
    // use LM for rescoring
    lm_->ComputeLMScore(lm_scale_, context_size, &cur);
  }

  std::vector<OfflineTransducerDecoderResult> unsorted_ans(batch_size);
  for (int32_t i = 0; i != batch_size; ++i) {
    Hypothesis hyp = cur[i].GetMostProbable(true);

    auto &r = unsorted_ans[packed_encoder_out.sorted_indexes[i]];

    // strip leading blanks
    r.tokens = {hyp.ys.begin() + context_size, hyp.ys.end()};
    r.timestamps = std::move(hyp.timestamps);
  }

  return unsorted_ans;
}

}  // namespace sherpa_onnx
