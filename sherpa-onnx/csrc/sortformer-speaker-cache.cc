// sherpa-onnx/csrc/sortformer-speaker-cache.cc
//
// Copyright (c)  2026  Silvio Tomatis

#include "sherpa-onnx/csrc/sortformer-speaker-cache.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Indices of the k largest values of a strided column. Ties go to the lower
// index. torch.topk leaves the order of exact ties unspecified.
std::vector<int32_t> TopK(const float *p, int32_t n, int32_t stride,
                          int32_t k) {
  k = std::min(k, n);
  std::vector<int32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [p, stride](int32_t a, int32_t b) {
                      float sa = p[static_cast<int64_t>(a) * stride];
                      float sb = p[static_cast<int64_t>(b) * stride];
                      if (sa != sb) {
                        return sa > sb;
                      }
                      return a < b;
                    });
  idx.resize(k);
  return idx;
}

}  // namespace

SortformerSpeakerCache::SortformerSpeakerCache(
    const SortformerSpeakerCacheConfig &config)
    : config_(config) {
  if (static_cast<int32_t>(config_.silence_embeds.size()) !=
      config_.hidden_size) {
    SHERPA_ONNX_LOGE("Expect %d silence embeddings. Given: %d",
                     config_.hidden_size,
                     static_cast<int32_t>(config_.silence_embeds.size()));
    SHERPA_ONNX_EXIT(-1);
  }

  // Share of the speaker cache every speaker is budgeted, excluding its
  // reserved silence slots
  int32_t budget = config_.speaker_cache_length / config_.num_speakers -
                   config_.num_silence_frames;

  min_positive_scores_ = static_cast<int32_t>(
      std::floor(budget * config_.min_positive_scores_rate));
  num_strong_boosted_frames_ =
      static_cast<int32_t>(std::floor(budget * config_.strong_boost_rate));
  num_weak_boosted_frames_ =
      static_cast<int32_t>(std::floor(budget * config_.weak_boost_rate));
}

std::vector<float> SortformerSpeakerCache::GetEmbeds() const {
  std::vector<float> ans;
  ans.reserve(cache_embeds_.size() + fifo_.size());
  ans.insert(ans.end(), cache_embeds_.begin(), cache_embeds_.end());
  ans.insert(ans.end(), fifo_.begin(), fifo_.end());
  return ans;
}

void SortformerSpeakerCache::Update(const float *chunk_embeds,
                                    int32_t num_chunk_frames,
                                    const float *probs,
                                    int32_t num_input_frames) {
  const int32_t hidden_size = config_.hidden_size;
  const int32_t num_speakers = config_.num_speakers;
  const int32_t factor = config_.subsampling_factor;

  const int32_t num_cache_frames = num_cache_frames_;
  const int32_t num_fifo_frames = num_fifo_frames_;

  if (num_input_frames <
      num_cache_frames + num_fifo_frames + num_chunk_frames) {
    SHERPA_ONNX_LOGE(
        "Expect at least %d input frames for %d cached and %d chunk frames. "
        "Given: %d",
        num_cache_frames + num_fifo_frames + num_chunk_frames,
        num_cache_frames + num_fifo_frames, num_chunk_frames, num_input_frames);
    SHERPA_ONNX_EXIT(-1);
  }

  fifo_.insert(
      fifo_.end(), chunk_embeds,
      chunk_embeds + static_cast<int64_t>(num_chunk_frames) * hidden_size);
  int32_t n = num_fifo_frames + num_chunk_frames;
  num_fifo_frames_ = n;

  // No frames move to the speaker cache until the FIFO overflows, then at
  // least speaker_cache_update_period oldest frames are moved.
  if (n <= config_.fifo_length) {
    return;
  }

  int32_t num_popped = std::min(
      std::max(config_.speaker_cache_update_period, n - config_.fifo_length),
      n);

  // Speaker probabilities at the encoder frame rate:
  // the average over its subsampling_factor rows
  auto pooled = [probs, factor, num_speakers](int32_t frame, int32_t s) {
    const float *p =
        probs + static_cast<int64_t>(frame) * factor * num_speakers;
    float sum = 0;
    for (int32_t i = 0; i != factor; ++i) {
      sum += p[i * num_speakers + s];
    }
    return sum / factor;
  };

  int32_t num_frames = num_cache_frames + num_popped;
  std::vector<float> new_probs(static_cast<int64_t>(num_frames) * num_speakers);

  if (is_compressed_) {
    // A compressed cache is out of order, so the probs stored alongside its
    // frames are the only ones
    std::copy(cache_probs_.begin(), cache_probs_.end(), new_probs.begin());
  } else {
    // An uncompressed cache still holds plain chunk frames, whose
    // probabilities this step re-estimates
    for (int32_t f = 0; f != num_cache_frames; ++f) {
      for (int32_t s = 0; s != num_speakers; ++s) {
        new_probs[f * num_speakers + s] = pooled(f, s);
      }
    }
  }

  // The popped frames follow the speaker cache in this step's input
  for (int32_t f = 0; f != num_popped; ++f) {
    for (int32_t s = 0; s != num_speakers; ++s) {
      new_probs[(num_cache_frames + f) * num_speakers + s] =
          pooled(num_cache_frames + f, s);
    }
  }

  cache_embeds_.insert(
      cache_embeds_.end(), fifo_.begin(),
      fifo_.begin() + static_cast<int64_t>(num_popped) * hidden_size);
  fifo_.erase(fifo_.begin(),
              fifo_.begin() + static_cast<int64_t>(num_popped) * hidden_size);
  num_fifo_frames_ = n - num_popped;

  if (num_frames > config_.speaker_cache_length) {
    Compress(&cache_embeds_, &new_probs);
    is_compressed_ = true;
    num_frames = config_.speaker_cache_length;
  }

  num_cache_frames_ = num_frames;
  cache_probs_ = std::move(new_probs);
}

std::vector<float> SortformerSpeakerCache::GetScores(
    const std::vector<float> &probs, int32_t num_frames) const {
  const int32_t num_speakers = config_.num_speakers;
  const float threshold = config_.prediction_score_threshold;
  const float log_half = std::log(0.5f);

  std::vector<float> scores(probs.size());
  for (int32_t f = 0; f != num_frames; ++f) {
    const float *p = probs.data() + f * num_speakers;
    float sum_log_complements = 0;
    for (int32_t s = 0; s != num_speakers; ++s) {
      sum_log_complements += std::log(std::max(1.0f - p[s], threshold));
    }

    for (int32_t s = 0; s != num_speakers; ++s) {
      float log_prob = std::log(std::max(p[s], threshold));
      float log_complement = std::log(std::max(1.0f - p[s], threshold));
      float score = log_prob - log_complement + sum_log_complements - log_half;
      scores[f * num_speakers + s] = p[s] > 0.5f ? score : kNegInf;
    }
  }

  // If a speaker has enough frames with a positive score, disable its
  // remaining speech frames, i.e., the ones overlapped with other speakers.
  for (int32_t s = 0; s != num_speakers; ++s) {
    int32_t num_positive = 0;
    for (int32_t f = 0; f != num_frames; ++f) {
      num_positive += scores[f * num_speakers + s] > 0;
    }

    if (num_positive < min_positive_scores_) {
      continue;
    }

    for (int32_t f = 0; f != num_frames; ++f) {
      float &score = scores[f * num_speakers + s];
      if (probs[f * num_speakers + s] > 0.5f && !(score > 0)) {
        score = kNegInf;
      }
    }
  }

  return scores;
}

void SortformerSpeakerCache::Compress(std::vector<float> *embeds,
                                      std::vector<float> *probs) const {
  const int32_t num_speakers = config_.num_speakers;
  const int32_t hidden_size = config_.hidden_size;
  const int32_t cache_length = config_.speaker_cache_length;
  const int32_t num_frames = static_cast<int32_t>(probs->size()) / num_speakers;
  const float log_half = std::log(0.5f);

  std::vector<float> scores = GetScores(*probs, num_frames);

  // Frames beyond the cache capacity are the ones popped from the FIFO
  for (int32_t f = cache_length; f < num_frames; ++f) {
    for (int32_t s = 0; s != num_speakers; ++s) {
      scores[f * num_speakers + s] += config_.latest_frames_score_boost;
    }
  }

  for (const auto &[k, boost] :
       {std::pair<int32_t, float>{num_strong_boosted_frames_, -2 * log_half},
        std::pair<int32_t, float>{num_weak_boosted_frames_, -log_half}}) {
    for (int32_t s = 0; s != num_speakers; ++s) {
      for (int32_t f : TopK(scores.data() + s, num_frames, num_speakers, k)) {
        scores[f * num_speakers + s] += boost;
      }
    }
  }

  // Speaker-major flattening, with num_silence_frames extra frames per
  // speaker whose score is +inf, so that they are always kept.
  const int32_t num_scored_frames = num_frames + config_.num_silence_frames;
  const int32_t num_flat = num_speakers * num_scored_frames;
  std::vector<float> flat(num_flat, std::numeric_limits<float>::infinity());
  for (int32_t s = 0; s != num_speakers; ++s) {
    for (int32_t f = 0; f != num_frames; ++f) {
      flat[s * num_scored_frames + f] = scores[f * num_speakers + s];
    }
  }

  std::vector<int32_t> picked = TopK(flat.data(), num_flat, 1, cache_length);

  // Disabled picks go to the end. The remaining ones are sorted, which keeps
  // the speakers grouped and their frames in the original order.
  const int32_t sentinel = num_flat;
  for (auto &i : picked) {
    if (flat[i] == kNegInf) {
      i = sentinel;
    }
  }
  std::sort(picked.begin(), picked.end());

  std::vector<float> new_embeds(static_cast<int64_t>(cache_length) *
                                hidden_size);
  std::vector<float> new_probs(static_cast<int64_t>(cache_length) *
                               num_speakers);

  for (int32_t j = 0; j != cache_length; ++j) {
    int32_t f = picked[j] == sentinel
                    ? num_frames
                    : std::min(picked[j] % num_scored_frames, num_frames);

    float *dst = new_embeds.data() + static_cast<int64_t>(j) * hidden_size;
    if (f == num_frames) {
      // silence slot or disabled pick: silence embedding, zero probabilities
      std::copy(config_.silence_embeds.begin(), config_.silence_embeds.end(),
                dst);
      continue;
    }

    const float *src = embeds->data() + static_cast<int64_t>(f) * hidden_size;
    std::copy(src, src + hidden_size, dst);
    std::copy(probs->begin() + f * num_speakers,
              probs->begin() + (f + 1) * num_speakers,
              new_probs.begin() + j * num_speakers);
  }

  *embeds = std::move(new_embeds);
  *probs = std::move(new_probs);
}

}  // namespace sherpa_onnx
