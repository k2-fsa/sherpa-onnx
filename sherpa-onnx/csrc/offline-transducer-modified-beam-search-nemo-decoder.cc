// sherpa-onnx/csrc/offline-transducer-modified-beam-search-nemo-decoder.cc
//
// Copyright (c)  2026  (authors: github.com/nefastosaturo, github.com/nullbio)

#include "sherpa-onnx/csrc/offline-transducer-modified-beam-search-nemo-decoder.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <tuple>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/packed-sequence.h"
#include "sherpa-onnx/csrc/slice.h"

namespace sherpa_onnx {

namespace {

// Helper structure to track hypothesis with decoder state
struct NeMoHypothesis {
  std::vector<int32_t> ys;          // token sequence (excluding initial blank)
  std::vector<int32_t> timestamps;  // timestamps for each token
  std::vector<int32_t> durations;   // durations for TDT
  std::vector<float> ys_probs;      // log probability for each token
  float log_prob = 0.0f;            // accumulated log probability
  std::vector<Ort::Value> decoder_states;       // RNN/LSTM states
  const ContextState *context_state = nullptr;  // context graph state
  OrtAllocator *allocator = nullptr;            // allocator for cloning states
  int32_t frame_offset = 0;  // current frame position for this hypothesis
  int32_t num_symbols =
      0;  // number of non-blank symbols emitted at current frame

  NeMoHypothesis() = default;

  // Copy constructor - needed for hypothesis expansion
  NeMoHypothesis(const NeMoHypothesis &other)
      : ys(other.ys),
        timestamps(other.timestamps),
        durations(other.durations),
        ys_probs(other.ys_probs),
        log_prob(other.log_prob),
        context_state(other.context_state),
        allocator(other.allocator),
        frame_offset(other.frame_offset),
        num_symbols(other.num_symbols) {
    // Deep copy of decoder states
    decoder_states.reserve(other.decoder_states.size());
    for (const auto &state : other.decoder_states) {
      decoder_states.push_back(Clone(allocator, &state));
    }
  }

  NeMoHypothesis &operator=(const NeMoHypothesis &other) {
    if (this != &other) {
      ys = other.ys;
      timestamps = other.timestamps;
      durations = other.durations;
      ys_probs = other.ys_probs;
      log_prob = other.log_prob;
      context_state = other.context_state;
      allocator = other.allocator;
      frame_offset = other.frame_offset;
      num_symbols = other.num_symbols;

      decoder_states.clear();
      decoder_states.reserve(other.decoder_states.size());
      for (const auto &state : other.decoder_states) {
        decoder_states.push_back(Clone(allocator, &state));
      }
    }
    return *this;
  }

  NeMoHypothesis(NeMoHypothesis &&) = default;
  NeMoHypothesis &operator=(NeMoHypothesis &&) = default;
};

}  // namespace

// Merge hypotheses with the same token sequence (log-add of their scores,
// keeping the state of the better one) and keep the best max_active_paths.
static void MergeAndPrune(std::vector<NeMoHypothesis> *hyps,
                          int32_t max_active_paths) {
  std::vector<NeMoHypothesis> merged;
  merged.reserve(hyps->size());
  LogAdd<float> log_add;
  for (auto &hyp : *hyps) {
    auto it = std::find_if(
        merged.begin(), merged.end(),
        [&hyp](const NeMoHypothesis &h) { return h.ys == hyp.ys; });
    if (it == merged.end()) {
      merged.push_back(std::move(hyp));
    } else if (hyp.log_prob > it->log_prob) {
      float log_prob = log_add(hyp.log_prob, it->log_prob);
      *it = std::move(hyp);
      it->log_prob = log_prob;
    } else {
      it->log_prob = log_add(it->log_prob, hyp.log_prob);
    }
  }

  int32_t keep =
      std::min(max_active_paths, static_cast<int32_t>(merged.size()));
  std::partial_sort(merged.begin(), merged.begin() + keep, merged.end(),
                    [](const NeMoHypothesis &a, const NeMoHypothesis &b) {
                      return a.log_prob > b.log_prob;
                    });
  merged.resize(keep);
  *hyps = std::move(merged);
}

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerModifiedBeamSearchNeMoDecoder::Decode(
    Ort::Value encoder_out, Ort::Value encoder_out_length,
    OfflineStream **ss /*= nullptr*/, int32_t n /*= 0*/) {
  auto encoder_shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(encoder_shape[0]);
  int32_t num_frames = static_cast<int32_t>(encoder_shape[1]);
  int32_t encoder_dim = static_cast<int32_t>(encoder_shape[2]);

  if (ss != nullptr) SHERPA_ONNX_CHECK_EQ(batch_size, n);

  int32_t vocab_size = model_->VocabSize();
  int32_t blank_id = vocab_size - 1;  // NeMo models have blank at the end
  int32_t max_symbols_per_frame = 10;

  // For TDT models, we need to know the number of duration bins
  // We'll detect this from the joiner output size on first run
  int32_t num_durations = 0;

  std::vector<ContextGraphPtr> context_graphs(batch_size, nullptr);

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  OrtAllocator *allocator = model_->Allocator();

  const float *encoder_data = encoder_out.GetTensorData<float>();

  // Get per-utterance lengths
  std::vector<int32_t> utterance_lengths(batch_size);
  auto length_type =
      encoder_out_length.GetTensorTypeAndShapeInfo().GetElementType();
  for (int32_t i = 0; i < batch_size; ++i) {
    utterance_lengths[i] =
        (length_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
            ? encoder_out_length.GetTensorData<int32_t>()[i]
            : static_cast<int32_t>(
                  encoder_out_length.GetTensorData<int64_t>()[i]);
  }

  std::vector<OfflineTransducerDecoderResult> results(batch_size);

  // Process each utterance independently (simpler for TDT with variable frame
  // positions)
  for (int32_t b = 0; b < batch_size; ++b) {
    const ContextState *context_state = nullptr;
    // Added to the score for each non-blank token: a positive value keeps
    // beam search from dropping speech after a pause. Set per stream with
    // stream.SetOption("length_reward", "0.5").
    float length_reward = 0.0f;
    if (ss != nullptr) {
      context_graphs[b] = ss[b]->GetContextGraph();
      if (context_graphs[b] != nullptr) {
        context_state = context_graphs[b]->Root();
      }
      length_reward = ss[b]->GetOptionFloat("length_reward", 0.0f);
      if (!std::isfinite(length_reward)) {
        SHERPA_ONNX_LOGE(
            "length_reward must be a finite number. Given: %f. Using 0",
            length_reward);
        length_reward = 0.0f;
      }
    }

    int32_t this_num_frames = utterance_lengths[b];
    const float *this_encoder = encoder_data + b * num_frames * encoder_dim;

    // Initialize with single hypothesis
    std::vector<NeMoHypothesis> cur_hyps;
    {
      NeMoHypothesis blank_hyp;
      blank_hyp.log_prob = 0.0f;
      blank_hyp.context_state = context_state;
      blank_hyp.allocator = allocator;
      blank_hyp.frame_offset = 0;
      blank_hyp.decoder_states = model_->GetDecoderInitStates(1);
      cur_hyps.push_back(std::move(blank_hyp));
    }

    // A TDT hypothesis jumps ahead by its predicted duration, so hypotheses
    // can be at different frames. Their scores are only comparable if they
    // have consumed the same frames: a hypothesis that is ahead has not yet
    // paid for the frames the others still have to consume. Keep one
    // bucket of hypotheses per frame and only compare (merge and prune)
    // hypotheses within a bucket.
    std::vector<std::vector<NeMoHypothesis>> pending(this_num_frames);
    std::vector<NeMoHypothesis> finished;

    auto place = [&](NeMoHypothesis &&hyp) {
      if (hyp.frame_offset >= this_num_frames) {
        finished.push_back(std::move(hyp));
      } else {
        pending[hyp.frame_offset].push_back(std::move(hyp));
      }
    };

    for (auto &hyp : cur_hyps) {
      place(std::move(hyp));
    }

    std::vector<float> selection_logits;
    int32_t t = 0;
    bool same_frame = false;
    while (true) {
      while (t < this_num_frames && pending[t].empty()) {
        ++t;
        same_frame = false;
      }

      if (t >= this_num_frames) {
        break;  // All hypotheses have finished
      }

      // Hypotheses that emit a token with duration 0 go back to pending[t]
      // and are expanded again in the next iteration.
      cur_hyps = std::move(pending[t]);
      pending[t].clear();
      MergeAndPrune(&cur_hyps, max_active_paths_);

      // A token extension adds a log-prob (<= 0) plus length_reward and
      // the context-graph score, so the early stop below is safe only when
      // neither of those can raise a hypothesis above its current score.
      bool can_gain = length_reward > 0 || context_graphs[b] != nullptr;
      if (same_frame && !is_tdt_ && !can_gain) {
        // Without durations every token stays on its frame. As in NeMo's
        // beam search, stop expanding on this frame once there are
        // max_active_paths hypotheses beyond it that are all better than
        // the best one left on it.
        float best = cur_hyps[0].log_prob;
        int32_t num_better = 0;
        for (int32_t k = t + 1;
             k < this_num_frames && num_better < max_active_paths_; ++k) {
          for (const auto &h : pending[k]) {
            num_better += h.log_prob > best;
          }
        }
        for (const auto &h : finished) {
          num_better += h.log_prob > best;
        }
        if (num_better >= max_active_paths_) {
          continue;
        }
      }
      same_frame = true;

      for (auto &hyp : cur_hyps) {
        if (hyp.num_symbols >= max_symbols_per_frame) {
          // Reached the per-frame symbol limit, force blank to advance frame
          NeMoHypothesis new_hyp;
          new_hyp.ys = hyp.ys;
          new_hyp.timestamps = hyp.timestamps;
          new_hyp.durations = hyp.durations;
          new_hyp.ys_probs = hyp.ys_probs;
          new_hyp.context_state = hyp.context_state;
          new_hyp.allocator = allocator;
          new_hyp.log_prob = hyp.log_prob;
          new_hyp.num_symbols = 0;
          new_hyp.decoder_states.reserve(hyp.decoder_states.size());
          for (const auto &state : hyp.decoder_states) {
            new_hyp.decoder_states.push_back(Clone(allocator, &state));
          }
          new_hyp.frame_offset = hyp.frame_offset + 1;
          place(std::move(new_hyp));
          continue;
        }

        // Get encoder output for this frame
        std::array<int64_t, 3> encoder_3d_shape{1, encoder_dim, 1};
        const float *frame_data = this_encoder + hyp.frame_offset * encoder_dim;

        Ort::Value encoder_out_frame = Ort::Value::CreateTensor(
            memory_info, const_cast<float *>(frame_data), encoder_dim,
            encoder_3d_shape.data(), encoder_3d_shape.size());

        // Prepare decoder input: use blank_id as initial token, then last
        // emitted token
        int32_t last_token = hyp.ys.empty() ? blank_id : hyp.ys.back();
        std::array<int64_t, 2> decoder_input_shape = {1, 1};
        std::vector<int32_t> decoder_input_data = {last_token};

        Ort::Value decoder_input = Ort::Value::CreateTensor(
            memory_info, decoder_input_data.data(), 1,
            decoder_input_shape.data(), decoder_input_shape.size());

        std::array<int64_t, 1> decoder_input_length_shape = {1};
        std::vector<int32_t> decoder_input_length_data = {1};

        Ort::Value decoder_input_length = Ort::Value::CreateTensor(
            memory_info, decoder_input_length_data.data(), 1,
            decoder_input_length_shape.data(),
            decoder_input_length_shape.size());

        // Clone decoder states for this expansion
        std::vector<Ort::Value> decoder_states_copy;
        decoder_states_copy.reserve(hyp.decoder_states.size());
        for (const auto &state : hyp.decoder_states) {
          decoder_states_copy.push_back(Clone(allocator, &state));
        }

        auto decoder_result = model_->RunDecoder(
            std::move(decoder_input), std::move(decoder_input_length),
            std::move(decoder_states_copy));

        Ort::Value decoder_out = std::move(decoder_result.first);
        std::vector<Ort::Value> next_states = std::move(decoder_result.second);

        // Run joiner
        Ort::Value logit =
            model_->RunJoiner(View(&encoder_out_frame), View(&decoder_out));

        auto logit_shape = logit.GetTensorTypeAndShapeInfo().GetShape();
        int32_t output_size = static_cast<int32_t>(logit_shape.back());

        float *p_logit = logit.GetTensorMutableData<float>();

        // Detect TDT mode from joiner output size
        if (is_tdt_ && num_durations == 0 && output_size > vocab_size) {
          num_durations = output_size - vocab_size;
        }

        // Split into token and duration logits for TDT
        int32_t token_vocab_size = is_tdt_ ? vocab_size : output_size;
        float *token_logits = p_logit;
        float *duration_logits = is_tdt_ ? (p_logit + vocab_size) : nullptr;

        // Apply blank penalty
        if (blank_penalty_ > 0.0f) {
          token_logits[blank_id] -= blank_penalty_;
        }

        // Compute log softmax for tokens only
        LogSoftmax(token_logits, token_vocab_size, 1);

        // Boost hotword continuations when selecting the top-k tokens so
        // they have a chance to be selected even if their base probability is
        // low. The boost is only used for the selection: the score of a
        // hypothesis gets the context graph score below, which is taken back
        // if the hotword is not completed.
        const float *selection = token_logits;
        if (context_graphs[b] != nullptr && hyp.context_state != nullptr) {
          selection_logits.assign(token_logits,
                                  token_logits + token_vocab_size);
          for (const auto &pair : hyp.context_state->next) {
            int32_t token_id = pair.first;
            if (token_id >= 0 && token_id < token_vocab_size) {
              selection_logits[token_id] += hotwords_score_;
            }
          }
          selection = selection_logits.data();
        }

        // Candidate durations: the top-k durations for TDT. Otherwise a token
        // stays on the current frame and blank advances by one frame.
        std::vector<int32_t> top_durations;
        if (is_tdt_ && duration_logits != nullptr && num_durations > 0) {
          LogSoftmax(duration_logits, num_durations, 1);
          top_durations = TopkIndex(duration_logits, num_durations,
                                    std::min(max_active_paths_, num_durations));
        }

        auto extend = [&](float log_prob) {
          NeMoHypothesis new_hyp;
          new_hyp.ys = hyp.ys;
          new_hyp.timestamps = hyp.timestamps;
          new_hyp.durations = hyp.durations;
          new_hyp.ys_probs = hyp.ys_probs;
          new_hyp.context_state = hyp.context_state;
          new_hyp.allocator = allocator;
          new_hyp.log_prob = log_prob;
          return new_hyp;
        };

        // Blank: keep the decoder state and advance by each candidate
        // duration. As in the greedy search, blank never has duration 0.
        std::vector<int32_t> blank_skips;
        for (int32_t d : top_durations) {
          if (d > 0) {
            blank_skips.push_back(d);
          }
        }
        if (blank_skips.empty()) {
          blank_skips.push_back(1);
        }

        for (int32_t d : blank_skips) {
          // For TDT: log P(blank, d) = log P(blank) + log P(d)
          float log_prob = hyp.log_prob + token_logits[blank_id];
          if (!top_durations.empty()) {
            log_prob += duration_logits[d];
          }

          NeMoHypothesis new_hyp = extend(log_prob);
          new_hyp.decoder_states.reserve(hyp.decoder_states.size());
          for (const auto &state : hyp.decoder_states) {
            new_hyp.decoder_states.push_back(Clone(allocator, &state));
          }
          new_hyp.frame_offset = hyp.frame_offset + d;
          new_hyp.num_symbols = 0;
          place(std::move(new_hyp));
        }

        // Non-blank: the top-k (token, duration) pairs by P(token) * P(d).
        // blank_id is the last token, so TopkIndex over the first blank_id
        // entries excludes it.
        std::vector<std::tuple<float, int32_t, int32_t>> pairs;
        for (int32_t token :
             TopkIndex(selection, blank_id, max_active_paths_)) {
          if (token == unk_id_) {
            continue;
          }
          if (top_durations.empty()) {
            pairs.emplace_back(selection[token], token, 0);
          }
          for (int32_t d : top_durations) {
            pairs.emplace_back(selection[token] + duration_logits[d], token, d);
          }
        }

        int32_t num_pairs =
            std::min(max_active_paths_, static_cast<int32_t>(pairs.size()));
        std::partial_sort(pairs.begin(), pairs.begin() + num_pairs, pairs.end(),
                          [](const auto &a, const auto &b) {
                            return std::get<0>(a) > std::get<0>(b);
                          });

        for (int32_t k = 0; k < num_pairs; ++k) {
          int32_t token = std::get<1>(pairs[k]);
          int32_t d = std::get<2>(pairs[k]);

          float log_prob = hyp.log_prob + token_logits[token] + length_reward;
          if (!top_durations.empty()) {
            log_prob += duration_logits[d];
          }

          NeMoHypothesis new_hyp = extend(log_prob);
          new_hyp.ys.push_back(token);
          new_hyp.timestamps.push_back(hyp.frame_offset);
          new_hyp.ys_probs.push_back(token_logits[token]);
          if (is_tdt_) {
            new_hyp.durations.push_back(d);
          }

          new_hyp.decoder_states.reserve(next_states.size());
          for (const auto &state : next_states) {
            new_hyp.decoder_states.push_back(Clone(allocator, &state));
          }

          // Update context graph
          if (context_graphs[b] != nullptr) {
            auto context_res = context_graphs[b]->ForwardOneStep(
                new_hyp.context_state, token, false);
            new_hyp.log_prob += std::get<0>(context_res);
            new_hyp.context_state = std::get<1>(context_res);
          }

          // With duration 0, more tokens can be emitted on the same frame
          new_hyp.frame_offset = hyp.frame_offset + d;
          new_hyp.num_symbols = d > 0 ? 0 : hyp.num_symbols + 1;
          place(std::move(new_hyp));
        }
      }
    }

    cur_hyps = std::move(finished);
    MergeAndPrune(&cur_hyps, max_active_paths_);

    // Finalize context biasing
    for (auto &hyp : cur_hyps) {
      if (context_graphs[b] != nullptr) {
        auto context_res = context_graphs[b]->Finalize(hyp.context_state);
        hyp.log_prob += context_res.first;
        hyp.context_state = context_res.second;
      }
    }

    // Find best hypothesis
    auto best_it =
        std::max_element(cur_hyps.begin(), cur_hyps.end(),
                         [](const NeMoHypothesis &a, const NeMoHypothesis &b) {
                           return a.log_prob < b.log_prob;
                         });

    if (best_it != cur_hyps.end()) {
      // Convert int32_t to int64_t for tokens
      results[b].tokens.assign(best_it->ys.begin(), best_it->ys.end());
      results[b].timestamps = best_it->timestamps;
      results[b].ys_log_probs = best_it->ys_probs;
      // Convert int32_t durations to float
      results[b].durations.reserve(best_it->durations.size());
      for (int32_t d : best_it->durations) {
        results[b].durations.push_back(static_cast<float>(d));
      }
    }
  }

  return results;
}

}  // namespace sherpa_onnx
