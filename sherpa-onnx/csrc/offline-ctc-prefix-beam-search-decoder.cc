// sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

static std::vector<Hypothesis> StepWorker(const float *p_log_probs,
                                          std::vector<Hypothesis> &hyps,
                                          int32_t blank_id, int32_t vocab_size,
                                          int32_t max_active_paths) {
  auto topk = TopkIndex(p_log_probs, vocab_size, max_active_paths);
  Hypotheses next_hyps;
  for (auto &hyp : hyps) {
    Hypothesis new_hyp = hyp;
    for (auto k : topk) {
      int32_t new_token = k;
      float log_prob = p_log_probs[k];
      if (new_token == blank_id) {
        // Case 0: *a + ε => *a
        //         *aε + ε => *a
        // Prefix does not change, update log_prob of blank
        new_hyp.log_prob_nb = -std::numeric_limits<float>::infinity();
        new_hyp.log_prob_b = hyp.LogProb(true) + log_prob;
        next_hyps.Add(std::move(new_hyp));
      } else if (hyp.ys.size() > 0 && hyp.ys.back() == new_token) {
        // Case 1: *a + a => *a
        // Prefix does not change, update log_prob of non_blank
        new_hyp.log_prob_nb = hyp.log_prob_nb + log_prob;
        new_hyp.log_prob_b = -std::numeric_limits<float>::infinity();

        next_hyps.Add(std::move(new_hyp));

        // Case 2: *aε + a => *aa
        // Prefix changes, update log_prob of blank
        new_hyp = hyp;
        new_hyp.ys.push_back(new_token);
        new_hyp.log_prob_nb = hyp.log_prob_b + log_prob;
        new_hyp.log_prob_b = -std::numeric_limits<float>::infinity();
        next_hyps.Add(std::move(new_hyp));
      } else {
        // Case 3: *a + b => *ab, *aε + b => *ab
        // Prefix changes, update log_prob of non_blank
        // Caution: DO NOT use append, as clone is shallow copy
        new_hyp.ys.push_back(new_token);
        new_hyp.log_prob_nb = hyp.LogProb(true) + log_prob;
        new_hyp.log_prob_b = -std::numeric_limits<float>::infinity();
        next_hyps.Add(std::move(new_hyp));
      }
    }
  }
  return next_hyps.GetTopK(max_active_paths, false, true);
}

std::vector<OfflineCtcDecoderResult> OfflineCtcPrefixBeamSearchDecoder::Decode(
    Ort::Value log_probs, Ort::Value log_probs_length) {
  std::vector<int64_t> shape = log_probs.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t num_frames = static_cast<int32_t>(shape[1]);
  int32_t vocab_size = static_cast<int32_t>(shape[2]);

  const int64_t *p_log_probs_length = log_probs_length.GetTensorData<int64_t>();

  std::vector<OfflineCtcDecoderResult> ans;
  ans.reserve(batch_size);

  std::vector<std::vector<Hypothesis>> cur;
  cur.reserve(batch_size);

  for (int32_t i = 0; i < batch_size; ++i) {
    cur.emplace_back(std::vector<Hypothesis>({Hypothesis()}));
  }

  for (int32_t t = 0; t < num_frames; ++t) {
    for (int32_t b = 0; b < batch_size; ++b) {
      if (t < p_log_probs_length[b]) {
        const float *p_log_probs = log_probs.GetTensorData<float>() +
                                   b * num_frames * vocab_size + t * vocab_size;
        cur[b] = StepWorker(p_log_probs, cur[b], blank_id_, vocab_size,
                            max_active_paths_);
      }
    }
  }

  for (int32_t b = 0; b != batch_size; ++b) {
    Hypotheses hyps(cur[b]);
    Hypothesis best_hyp = hyps.GetMostProbable(false, true);
    OfflineCtcDecoderResult r;
    r.tokens = best_hyp.ys;
    ans.push_back(std::move(r));
  }
  return ans;
}

}  // namespace sherpa_onnx
