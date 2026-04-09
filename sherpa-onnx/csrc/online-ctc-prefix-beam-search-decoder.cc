// sherpa-onnx/csrc/online-ctc-prefix-beam-search-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-ctc-prefix-beam-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

static std::vector<Hypothesis> StepWorker(const float *p_log_probs,
                                          std::vector<Hypothesis> &hyps,
                                          int32_t blank_id, int32_t vocab_size,
                                          int32_t max_active_paths,
                                          const ContextGraph *context_graph) {
  auto topk = TopkIndex(p_log_probs, vocab_size, max_active_paths);
  Hypotheses next_hyps;
  for (auto &hyp : hyps) {
    for (auto k : topk) {
      Hypothesis new_hyp = hyp;
      int32_t new_token = k;
      float log_prob = p_log_probs[k];
      bool update_prefix = false;
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
        update_prefix = true;
      } else {
        // Case 3: *a + b => *ab, *aε + b => *ab
        // Prefix changes, update log_prob of non_blank
        new_hyp.ys.push_back(new_token);
        new_hyp.log_prob_nb = hyp.LogProb(true) + log_prob;
        new_hyp.log_prob_b = -std::numeric_limits<float>::infinity();
        update_prefix = true;
      }

      if (update_prefix) {
        float lm_log_prob = hyp.lm_log_prob;
        if (context_graph != nullptr && hyp.context_state != nullptr) {
          auto context_res =
              context_graph->ForwardOneStep(hyp.context_state, new_token);
          lm_log_prob = lm_log_prob + std::get<0>(context_res);
          new_hyp.context_state = std::get<1>(context_res);
        }
        new_hyp.lm_log_prob = lm_log_prob;
        next_hyps.Add(std::move(new_hyp));
      }
    }
  }
  return next_hyps.GetTopK(max_active_paths, false, true);
}

void OnlineCtcPrefixBeamSearchDecoder::Decode(
    const float *log_probs, int32_t batch_size, int32_t num_frames,
    int32_t vocab_size, std::vector<OnlineCtcDecoderResult> *results,
    OnlineStream **ss /*= nullptr*/, int32_t /*n = 0*/) {
  if (batch_size != static_cast<int32_t>(results->size())) {
    SHERPA_ONNX_LOGE("Size mismatch! batch_size %d, results.size(): %d",
                     batch_size, static_cast<int32_t>(results->size()));
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<ContextGraphPtr> context_graphs(batch_size, nullptr);

  // Initialize hypotheses for each utterance in the batch
  std::vector<std::vector<Hypothesis>> cur;
  cur.reserve(batch_size);

  for (int32_t b = 0; b < batch_size; ++b) {
    auto &r = (*results)[b];
    if (r.hyps.Size() == 0) {
      // First chunk: initialize with an empty hypothesis
      const ContextState *context_state = nullptr;
      if (ss != nullptr && ss[b] != nullptr) {
        context_graphs[b] = ss[b]->GetContextGraph();
        if (context_graphs[b] != nullptr) {
          context_state = context_graphs[b]->Root();
        }
      }
      Hypothesis hyp(context_state);
      cur.emplace_back(std::vector<Hypothesis>({hyp}));
    } else {
      // Subsequent chunks: restore hypotheses from the result
      if (ss != nullptr && ss[b] != nullptr) {
        context_graphs[b] = ss[b]->GetContextGraph();
      }
      cur.push_back(r.hyps.Vec());
    }
  }

  // Process each frame in the chunk
  for (int32_t t = 0; t < num_frames; ++t) {
    for (int32_t b = 0; b < batch_size; ++b) {
      const float *p = log_probs + b * num_frames * vocab_size + t * vocab_size;
      cur[b] = StepWorker(p, cur[b], blank_id_, vocab_size,
                          max_active_paths_, context_graphs[b].get());
    }
  }

  // Update results
  for (int32_t b = 0; b < batch_size; ++b) {
    auto &r = (*results)[b];

    // Save hypotheses for the next chunk
    r.hyps = Hypotheses(cur[b]);

    // Get the best hypothesis for the current result
    Hypothesis best_hyp = r.hyps.GetMostProbable(false, true);

    r.tokens = best_hyp.ys;
    r.frame_offset += num_frames;

    // Count trailing blanks for endpointing
    if (!best_hyp.ys.empty()) {
      r.num_trailing_blanks = 0;
    }
  }
}

}  // namespace sherpa_onnx
