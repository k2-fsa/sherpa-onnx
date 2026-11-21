// sherpa-onnx/csrc/online-transducer-modified-beam-search-nemo-decoder.cc
//
// Copyright (c)  2026  Joseph Mills (github.com/josephomills)

#include "sherpa-onnx/csrc/online-transducer-modified-beam-search-nemo-decoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-model.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-parakeet-unified-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

namespace {

// A hypothesis plus its chunk-local decoding position. Only the Hypothesis
// part is persisted across chunks; at a chunk boundary every candidate has
// consumed all frames of the chunk, so `frame` and `num_symbols` restart
// from 0 with the next chunk.
struct Candidate {
  Hypothesis hyp;
  int32_t frame = 0;        // chunk-local frame index
  int32_t num_symbols = 0;  // non-blank symbols emitted at the current frame
};

Ort::Value BuildDecoderInput(int32_t token, OrtAllocator *allocator) {
  std::array<int64_t, 2> shape{1, 1};

  Ort::Value decoder_input =
      Ort::Value::CreateTensor<int32_t>(allocator, shape.data(), shape.size());

  int32_t *p = decoder_input.GetTensorMutableData<int32_t>();

  p[0] = token;

  return decoder_input;
}

template <typename Model>
void EnsurePredictor(const Model *model, int32_t blank_id, Hypothesis *hyp) {
  if (hyp->nemo_decoder_out.value) {
    return;
  }

  // Non-blank extensions consume their last token only after pruning.
  int32_t last_token =
      hyp->ys.empty() ? blank_id : static_cast<int32_t>(hyp->ys.back());
  auto decoder_result =
      model->RunDecoder(BuildDecoderInput(last_token, model->Allocator()),
                        Convert(std::move(hyp->nemo_decoder_states)));
  hyp->nemo_decoder_out = std::move(decoder_result.first);
  hyp->nemo_decoder_states = Convert(std::move(decoder_result.second));
}

template <typename Model>
void DecodeOne(const float *encoder_out, int32_t num_rows, int32_t num_cols,
               const Model *model, int32_t max_active_paths,
               float blank_penalty, OnlineStream *s) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  constexpr bool kUnified =
      std::is_same_v<Model, OnlineTransducerNeMoParakeetUnifiedModel>;

  OrtAllocator *allocator = model->Allocator();

  int32_t vocab_size = model->VocabSize();
  if (vocab_size <= 0) {
    SHERPA_ONNX_LOGE("Invalid NeMo transducer vocabulary size.");
    SHERPA_ONNX_EXIT(-1);
  }
  int32_t blank_id = vocab_size - 1;
  int32_t max_symbols_per_frame = 10;

  auto &r = s->GetResult();
  const auto &context_graph = s->GetContextGraph();
  int32_t frame_offset = r.frame_offset;
  if (frame_offset < 0 ||
      num_rows > std::numeric_limits<int32_t>::max() - frame_offset) {
    SHERPA_ONNX_LOGE("NeMo transducer frame offset is out of range.");
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<Candidate> cur;

  if (r.hyps.Size() == 0) {
    // First chunk of this stream (or first after Reset)
    Candidate c;
    c.hyp.log_prob = 0;
    c.hyp.context_state =
        context_graph != nullptr ? context_graph->Root() : nullptr;
    c.hyp.nemo_decoder_states = Convert(model->GetDecoderInitStates());
    if constexpr (kUnified) {
      auto &committed_out = s->GetNeMoDecoderOut();
      if (committed_out) {
        // Reset committed the previous result. Decode a fresh suffix from
        // its predictor state, which is independent of the visible tokens.
        c.hyp.nemo_decoder_out = Clone(allocator, &committed_out);
        c.hyp.nemo_decoder_states.clear();
        for (const auto &state : s->GetNeMoDecoderStates()) {
          c.hyp.nemo_decoder_states.emplace_back(Clone(allocator, &state));
        }
      }
    }
    cur.push_back(std::move(c));
  } else {
    cur.reserve(r.hyps.Size());
    for (auto &p : r.hyps) {
      Candidate c;
      c.hyp = std::move(p.second);
      cur.push_back(std::move(c));
    }
    r.hyps.Clear();
  }

  while (true) {
    // Hypotheses can be at different frames since a non-blank emission does
    // not advance the frame. Always expand the ones that are furthest behind.
    int32_t min_frame = num_rows;
    for (const auto &c : cur) {
      min_frame = std::min(min_frame, c.frame);
    }

    if (min_frame >= num_rows) {
      break;  // all hypotheses have consumed the whole chunk
    }

    std::vector<std::pair<double, Candidate>> all_candidates;

    for (auto &c : cur) {
      if (c.frame > min_frame) {
        // This hypothesis is ahead; keep it as-is
        double log_prob = c.hyp.log_prob;
        all_candidates.emplace_back(log_prob, std::move(c));
        continue;
      }

      if (c.num_symbols >= max_symbols_per_frame) {
        // Reached the per-frame symbol limit; force advancing to the next
        // frame without emitting a token
        c.frame += 1;
        c.num_symbols = 0;
        double log_prob = c.hyp.log_prob;
        all_candidates.emplace_back(log_prob, std::move(c));
        continue;
      }

      EnsurePredictor(model, blank_id, &c.hyp);

      std::array<int64_t, 3> encoder_shape{1, num_cols, 1};

      Ort::Value cur_encoder_out = Ort::Value::CreateTensor(
          memory_info,
          const_cast<float *>(encoder_out) +
              static_cast<size_t>(c.frame) * num_cols,
          num_cols, encoder_shape.data(), encoder_shape.size());

      Ort::Value logit = model->RunJoiner(
          View(&cur_encoder_out), View(&c.hyp.nemo_decoder_out.value));

      if constexpr (kUnified) {
        const auto info = logit.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetElementCount() != static_cast<size_t>(vocab_size)) {
          SHERPA_ONNX_LOGE("Invalid Parakeet Unified joiner output.");
          SHERPA_ONNX_EXIT(-1);
        }
      }
      float *p_logit = logit.GetTensorMutableData<float>();
      if (blank_penalty > 0) {
        p_logit[blank_id] -= blank_penalty;
      }

      int32_t top_token = -1;
      if constexpr (kUnified) {
        if (max_active_paths == 1) {
          // Match greedy selection even if log-softmax rounds two scores
          // to the same value. Use the normalized score for accumulation.
          top_token = MaxElementIndex(p_logit, vocab_size);
        }
      }

      LogSoftmax(p_logit, vocab_size);
      if constexpr (kUnified) {
        if (std::any_of(p_logit, p_logit + vocab_size,
                        [](float score) { return std::isnan(score); })) {
          // An unusable row cannot rank tokens. Keep this predictor and
          // advance as a blank so one bad frame does not stop other streams.
          ++c.frame;
          c.num_symbols = 0;
          ++c.hyp.num_trailing_blanks;
          all_candidates.emplace_back(c.hyp.log_prob, std::move(c));
          continue;
        }
      }

      // Boost hotword continuations by their context-graph transition score
      // before the top-k selection, so that they have a chance to be selected
      // even if their base probability is low. The boost is used for ranking
      // only: the original values are restored right after, so path scores
      // and ys_probs stay purely acoustic and the context-graph score is
      // added exactly once via ForwardOneStep().
      std::vector<std::pair<int32_t, float>> boosted;
      if (context_graph != nullptr && c.hyp.context_state != nullptr) {
        for (const auto &pair : c.hyp.context_state->next) {
          int32_t token_id = pair.first;
          if (token_id >= 0 && token_id < vocab_size) {
            boosted.emplace_back(token_id, p_logit[token_id]);
            p_logit[token_id] += pair.second->token_score;
          }
        }
      }

      auto top_k = top_token >= 0
                       ? std::vector<int32_t>{top_token}
                       : TopkIndex(p_logit, vocab_size, max_active_paths);

      for (const auto &b : boosted) {
        p_logit[b.first] = b.second;
      }

      for (int32_t token : top_k) {
        Candidate nc;
        nc.hyp.ys = c.hyp.ys;
        nc.hyp.timestamps = c.hyp.timestamps;
        nc.hyp.ys_probs = c.hyp.ys_probs;
        nc.hyp.context_scores = c.hyp.context_scores;
        nc.hyp.context_state = c.hyp.context_state;
        nc.hyp.num_trailing_blanks = c.hyp.num_trailing_blanks;
        nc.hyp.log_prob = c.hyp.log_prob + p_logit[token];

        if (token == blank_id) {
          // Keep the decoder output and states, advance the frame
          nc.hyp.nemo_decoder_states = c.hyp.nemo_decoder_states;
          nc.hyp.nemo_decoder_out = c.hyp.nemo_decoder_out;
          nc.frame = c.frame + 1;
          nc.num_symbols = 0;
          nc.hyp.num_trailing_blanks += 1;
        } else {
          nc.hyp.ys.push_back(token);
          nc.hyp.timestamps.push_back(c.frame + frame_offset);
          nc.hyp.ys_probs.push_back(p_logit[token]);
          nc.hyp.num_trailing_blanks = 0;

          // The states stay one token behind; the new token is consumed
          // lazily at this candidate's next expansion (nemo_decoder_out is
          // left unset), and only if the candidate survives pruning.
          nc.hyp.nemo_decoder_states = c.hyp.nemo_decoder_states;

          // Stay on the same frame to allow emitting more tokens
          nc.frame = c.frame;
          nc.num_symbols = c.num_symbols + 1;

          if (context_graph != nullptr) {
            auto context_res = context_graph->ForwardOneStep(
                nc.hyp.context_state, token, false /*strict mode*/);
            nc.hyp.log_prob += std::get<0>(context_res);
            nc.hyp.context_scores.push_back(std::get<0>(context_res));
            nc.hyp.context_state = std::get<1>(context_res);
          }
        }

        double log_prob = nc.hyp.log_prob;
        all_candidates.emplace_back(log_prob, std::move(nc));
      }
    }

    // Recombine candidates that have an identical token sequence at the
    // same frame with the same per-frame symbol count, keeping only the
    // best-scoring alignment (Viterbi-style). Their decoder states and
    // context states are identical, since both are deterministic functions
    // of the emitted tokens. Without recombination, the beam fills up with
    // alignments of one and the same token sequence, and paths that emit
    // tokens in low-confidence regions get pruned in favor of blank-only
    // paths, which causes deletions on long audio. Keeping the maximum
    // instead of summing (marginalizing over emission positions) matches
    // the offline decoder and greedy search; summing would let a weak but
    // consistent token accumulate mass across all possible emission
    // positions and beat silence on noise-only audio.
    {
      std::unordered_map<std::string, size_t> index;
      std::vector<std::pair<double, Candidate>> merged;
      merged.reserve(all_candidates.size());

      for (auto &p : all_candidates) {
        std::string key = p.second.hyp.Key() + "#" +
                          std::to_string(p.second.frame) + "#" +
                          std::to_string(p.second.num_symbols);
        auto it = index.find(key);
        if (it == index.end()) {
          index[key] = merged.size();
          merged.push_back(std::move(p));
          continue;
        }

        auto &dst = merged[it->second];
        if (p.second.hyp.log_prob > dst.second.hyp.log_prob) {
          dst = std::move(p);
        }
      }

      all_candidates = std::move(merged);
    }

    // Keep the top max_active_paths candidates
    int32_t keep = std::min(
        max_active_paths, static_cast<int32_t>(all_candidates.size()));

    std::partial_sort(
        all_candidates.begin(), all_candidates.begin() + keep,
        all_candidates.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });

    cur.clear();
    cur.reserve(keep);
    for (int32_t k = 0; k != keep; ++k) {
      cur.push_back(std::move(all_candidates[k].second));
    }
  }

  // Persist the hypotheses for the next chunk. Hypotheses with an identical
  // token sequence are merged; their decoder states are identical since the
  // prediction network is a deterministic function of the emitted tokens.
  Hypotheses hyps;
  for (auto &c : cur) {
    hyps.Add(std::move(c.hyp));
  }

  // Select the best hypothesis by raw score, matching the offline decoder.
  // Length normalization must not be used here: a blank-only hypothesis
  // (valid for silent audio) has an empty token sequence, and dividing
  // token-emitting paths by their length lets weak tokens beat silence on
  // noise-only audio.
  Hypothesis best = hyps.GetMostProbable(false /*length_norm*/);

  if constexpr (kUnified) {
    // Publish the predictor from the same path as the visible result. It
    // survives Reset(), while the segment's hypotheses are cleared.
    if (!best.nemo_decoder_out.value) {
      // A forced advance at the symbol limit can leave the last token
      // pending. Cache its output on the retained path as well.
      EnsurePredictor(model, blank_id, &best);
      const auto key = best.Key();
      for (auto &entry : hyps) {
        if (entry.first == key) {
          entry.second.nemo_decoder_out = best.nemo_decoder_out;
          entry.second.nemo_decoder_states = best.nemo_decoder_states;
          break;
        }
      }
    }
    s->SetNeMoDecoderOut(std::move(best.nemo_decoder_out.value));
    s->SetNeMoDecoderStates(Convert(std::move(best.nemo_decoder_states)));
  }

  r.hyps = std::move(hyps);
  r.tokens = std::move(best.ys);
  r.timestamps = std::move(best.timestamps);
  r.ys_probs = std::move(best.ys_probs);
  r.context_scores = std::move(best.context_scores);
  r.num_trailing_blanks = best.num_trailing_blanks;
  r.frame_offset += num_rows;
}

template <typename Model>
void DecodeBatch(const Ort::Value &encoder_out, OnlineStream **ss, int32_t n,
                 const Model *model, int32_t max_active_paths,
                 float blank_penalty) {
  const auto info = encoder_out.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  if (shape.size() != 3 || n < 0 || shape[0] != n || shape[1] < 0 ||
      shape[1] > std::numeric_limits<int32_t>::max() || shape[2] <= 0 ||
      shape[2] > std::numeric_limits<int32_t>::max() ||
      info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    SHERPA_ONNX_LOGE("Invalid NeMo transducer encoder output.");
    SHERPA_ONNX_EXIT(-1);
  }
  if (n == 0 || shape[1] == 0) {
    return;
  }

  const auto dim1 = static_cast<int32_t>(shape[1]);  // T
  const auto dim2 = static_cast<int32_t>(shape[2]);  // encoder_out_dim
  const auto per_stream = info.GetElementCount() / static_cast<size_t>(n);
  if (per_stream / static_cast<size_t>(dim2) != static_cast<size_t>(dim1) ||
      per_stream % static_cast<size_t>(dim2) != 0) {
    SHERPA_ONNX_LOGE("Invalid NeMo transducer encoder output size.");
    SHERPA_ONNX_EXIT(-1);
  }

  const float *p = encoder_out.GetTensorData<float>();

  for (int32_t i = 0; i != n; ++i) {
    const float *this_p = p + per_stream * i;

    DecodeOne(this_p, dim1, dim2, model, max_active_paths, blank_penalty,
              ss[i]);
  }
}

}  // namespace

void OnlineTransducerModifiedBeamSearchNeMoDecoder::Decode(
    Ort::Value encoder_out, OnlineStream **ss, int32_t n) const {
  std::visit(
      [&](const auto *model) {
        DecodeBatch(encoder_out, ss, n, model, max_active_paths_,
                    blank_penalty_);
      },
      model_);
}

}  // namespace sherpa_onnx
