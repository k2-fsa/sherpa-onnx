// sherpa-onnx/csrc/offline-transducer-modified-beam-search-nemo-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-transducer-modified-beam-search-nemo-decoder.h"

#include <algorithm>
#include <deque>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/packed-sequence.h"
#include "sherpa-onnx/csrc/slice.h"

namespace sherpa_onnx {

// Helper structure to track hypothesis with decoder state
struct NeMoHypothesis {
  std::vector<int32_t> ys;           // token sequence (excluding initial blank)
  std::vector<int32_t> timestamps;   // timestamps for each token
  std::vector<int32_t> durations;    // durations for TDT
  std::vector<float> ys_probs;       // log probability for each token
  float log_prob;                     // accumulated log probability
  std::vector<Ort::Value> decoder_states;  // RNN/LSTM states
  const ContextState *context_state;  // context graph state
  OrtAllocator *allocator;            // allocator for cloning states
  int32_t frame_offset;               // current frame position for this hypothesis

  NeMoHypothesis() : log_prob(0.0f), context_state(nullptr), allocator(nullptr), frame_offset(0) {}

  // Copy constructor - needed for hypothesis expansion
  NeMoHypothesis(const NeMoHypothesis &other)
      : ys(other.ys),
        timestamps(other.timestamps),
        durations(other.durations),
        ys_probs(other.ys_probs),
        log_prob(other.log_prob),
        context_state(other.context_state),
        allocator(other.allocator),
        frame_offset(other.frame_offset) {
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
  auto length_type = encoder_out_length.GetTensorTypeAndShapeInfo().GetElementType();
  for (int32_t i = 0; i < batch_size; ++i) {
    utterance_lengths[i] = (length_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
        ? encoder_out_length.GetTensorData<int32_t>()[i]
        : static_cast<int32_t>(encoder_out_length.GetTensorData<int64_t>()[i]);
  }

  std::vector<OfflineTransducerDecoderResult> results(batch_size);

  // Process each utterance independently (simpler for TDT with variable frame positions)
  for (int32_t b = 0; b < batch_size; ++b) {
    const ContextState *context_state = nullptr;
    if (ss != nullptr) {
      context_graphs[b] = ss[b]->GetContextGraph();
      if (context_graphs[b] != nullptr) {
        context_state = context_graphs[b]->Root();
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

    // Process until all hypotheses have finished
    while (true) {
      // Find minimum frame offset among active hypotheses
      int32_t min_frame = this_num_frames;
      for (const auto &hyp : cur_hyps) {
        if (hyp.frame_offset < min_frame) {
          min_frame = hyp.frame_offset;
        }
      }

      if (min_frame >= this_num_frames) {
        break;  // All hypotheses have finished
      }

      // Process hypotheses at the minimum frame
      std::vector<std::pair<float, NeMoHypothesis>> all_candidates;

      for (auto &hyp : cur_hyps) {
        if (hyp.frame_offset > min_frame) {
          // This hypothesis is ahead, keep it as-is
          all_candidates.emplace_back(hyp.log_prob, std::move(hyp));
          continue;
        }

        // Get encoder output for this frame
        std::array<int64_t, 3> encoder_3d_shape{1, encoder_dim, 1};
        const float *frame_data = this_encoder + hyp.frame_offset * encoder_dim;

        Ort::Value encoder_out_frame = Ort::Value::CreateTensor(
            memory_info, const_cast<float*>(frame_data), encoder_dim,
            encoder_3d_shape.data(), encoder_3d_shape.size());

        // Prepare decoder input: use blank_id as initial token, then last emitted token
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
            decoder_input_length_shape.data(), decoder_input_length_shape.size());

        // Clone decoder states for this expansion
        std::vector<Ort::Value> decoder_states_copy;
        decoder_states_copy.reserve(hyp.decoder_states.size());
        for (const auto &state : hyp.decoder_states) {
          decoder_states_copy.push_back(Clone(allocator, &state));
        }

        auto decoder_result = model_->RunDecoder(
            std::move(decoder_input),
            std::move(decoder_input_length),
            std::move(decoder_states_copy));

        Ort::Value decoder_out = std::move(decoder_result.first);
        std::vector<Ort::Value> next_states = std::move(decoder_result.second);

        // Run joiner
        Ort::Value logit = model_->RunJoiner(
            View(&encoder_out_frame),
            View(&decoder_out));

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

        // Apply context boosting BEFORE top-k selection so hotword tokens
        // have a chance to be selected even if their base probability is low
        if (context_graphs[b] != nullptr && hyp.context_state != nullptr) {
          for (const auto &pair : hyp.context_state->next) {
            int32_t token_id = pair.first;
            if (token_id >= 0 && token_id < token_vocab_size) {
              token_logits[token_id] += hotwords_score_;
            }
          }
        }

        auto top_k_tokens = TopkIndex(token_logits, token_vocab_size, max_active_paths_);

        // Determine duration/skip for TDT
        int32_t predicted_skip = 1;  // Default: advance by 1 frame
        if (is_tdt_ && duration_logits != nullptr && num_durations > 0) {
          // Find best duration
          predicted_skip = static_cast<int32_t>(std::distance(
              duration_logits,
              std::max_element(duration_logits, duration_logits + num_durations)));
        }

        // Create candidate hypotheses
        for (int32_t idx : top_k_tokens) {
          int32_t token = idx;
          float token_log_prob = token_logits[token] + hyp.log_prob;

          NeMoHypothesis new_hyp;
          new_hyp.ys = hyp.ys;
          new_hyp.timestamps = hyp.timestamps;
          new_hyp.durations = hyp.durations;
          new_hyp.ys_probs = hyp.ys_probs;
          new_hyp.context_state = hyp.context_state;
          new_hyp.allocator = allocator;
          new_hyp.log_prob = token_log_prob;

          float context_score = 0.0f;

          if (token == blank_id || token == unk_id_) {
            // Blank or unk: keep decoder state, advance frame
            new_hyp.decoder_states.reserve(hyp.decoder_states.size());
            for (const auto &state : hyp.decoder_states) {
              new_hyp.decoder_states.push_back(Clone(allocator, &state));
            }
            // For blank/unk in TDT, always advance by at least 1
            new_hyp.frame_offset = hyp.frame_offset + std::max(1, predicted_skip);
          } else {
            // Non-blank: add token, use new decoder state
            new_hyp.ys.push_back(token);
            new_hyp.timestamps.push_back(hyp.frame_offset);
            new_hyp.ys_probs.push_back(token_logits[token]);
            if (is_tdt_) {
              new_hyp.durations.push_back(predicted_skip);
            }

            new_hyp.decoder_states.reserve(next_states.size());
            for (const auto &state : next_states) {
              new_hyp.decoder_states.push_back(Clone(allocator, &state));
            }

            // For non-blank in TDT, advance by predicted duration (can be 0 to emit more tokens)
            // For non-TDT, stay on same frame to allow more tokens
            if (is_tdt_) {
              new_hyp.frame_offset = hyp.frame_offset + predicted_skip;
            } else {
              new_hyp.frame_offset = hyp.frame_offset;
            }

            // Update context graph
            if (context_graphs[b] != nullptr) {
              auto context_res = context_graphs[b]->ForwardOneStep(
                  new_hyp.context_state, token, false);
              context_score = std::get<0>(context_res);
              new_hyp.context_state = std::get<1>(context_res);
            }
            new_hyp.log_prob += context_score;
          }

          all_candidates.emplace_back(new_hyp.log_prob, std::move(new_hyp));
        }
      }

      // Keep top-k hypotheses
      if (all_candidates.empty()) {
        break;
      }

      std::partial_sort(
          all_candidates.begin(),
          all_candidates.begin() + std::min(max_active_paths_,
                                           static_cast<int32_t>(all_candidates.size())),
          all_candidates.end(),
          [](const auto &a, const auto &b) { return a.first > b.first; });

      int32_t keep = std::min(max_active_paths_,
                             static_cast<int32_t>(all_candidates.size()));
      cur_hyps.clear();
      cur_hyps.reserve(keep);
      for (int32_t k = 0; k < keep; ++k) {
        cur_hyps.push_back(std::move(all_candidates[k].second));
      }
    }

    // Finalize context biasing
    for (auto &hyp : cur_hyps) {
      if (context_graphs[b] != nullptr) {
        auto context_res = context_graphs[b]->Finalize(hyp.context_state);
        hyp.log_prob += context_res.first;
        hyp.context_state = context_res.second;
      }
    }

    // Find best hypothesis
    auto best_it = std::max_element(
        cur_hyps.begin(), cur_hyps.end(),
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
