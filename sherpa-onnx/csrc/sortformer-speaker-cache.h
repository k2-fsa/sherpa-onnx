// sherpa-onnx/csrc/sortformer-speaker-cache.h
//
// Copyright (c)  2026  Silvio Tomatis

#ifndef SHERPA_ONNX_CSRC_SORTFORMER_SPEAKER_CACHE_H_
#define SHERPA_ONNX_CSRC_SORTFORMER_SPEAKER_CACHE_H_

#include <cstdint>
#include <vector>

namespace sherpa_onnx {

struct SortformerSpeakerCacheConfig {
  int32_t num_speakers = 8;
  int32_t hidden_size = 512;

  // Number of 10 ms output frames per encoder frame
  int32_t subsampling_factor = 8;

  // Capacity of the FIFO queue of the most recent encoder frames
  int32_t fifo_length = 40;

  // Minimum number of frames moved from the FIFO queue to the speaker cache
  // when the queue overflows
  int32_t speaker_cache_update_period = 300;

  int32_t speaker_cache_length = 264;

  // Number of slots per speaker reserved for the silence embedding
  int32_t num_silence_frames = 1;

  float prediction_score_threshold = 0.25f;
  float latest_frames_score_boost = 0.05f;
  float min_positive_scores_rate = 0.5f;
  float strong_boost_rate = 0.75f;
  float weak_boost_rate = 1.5f;

  // Learned silence embedding, of size hidden_size
  std::vector<float> silence_embeds;
};

// Streaming state of the Sortformer diarization model: the Arrival-Order
// Speaker Cache (AOSC) and the FIFO queue of the most recent encoder frames.
//
// It follows Nemotron3DiarizationSpeakerCache from Hugging Face transformers
// and the synchronous streaming_update() of NeMo's SortformerModules.
// See also src/asr/diar/aosc_state.cpp from
// https://github.com/NVIDIA/NeMo-Speech.cpp
class SortformerSpeakerCache {
 public:
  explicit SortformerSpeakerCache(const SortformerSpeakerCacheConfig &config);

  // Number of embedding frames returned by GetEmbeds()
  int32_t NumCachedFrames() const {
    return num_cache_frames_ + num_fifo_frames_;
  }

  int32_t NumSpeakerCacheFrames() const { return num_cache_frames_; }
  int32_t NumFifoFrames() const { return num_fifo_frames_; }

  // Return the speaker cache followed by the FIFO queue. It is a row-major
  // matrix of shape (NumCachedFrames(), hidden_size).
  std::vector<float> GetEmbeds() const;

  /** Push a processed chunk to the FIFO queue, moving its oldest frames to
   *  the speaker cache when it overflows.
   *
   * @param chunk_embeds Row-major (num_chunk_frames, hidden_size) embeddings
   *                     of the chunk, without its right context.
   * @param num_chunk_frames Number of frames of the chunk.
   * @param probs Row-major (num_input_frames * subsampling_factor,
   *              num_speakers) speaker probabilities of this step. Their
   *              rows are for the frames returned by GetEmbeds() before this
   *              call, followed by the chunk and its right context.
   * @param num_input_frames Number of encoder frames of this step.
   */
  void Update(const float *chunk_embeds, int32_t num_chunk_frames,
              const float *probs, int32_t num_input_frames);

 private:
  // Keep speaker_cache_length frames, grouped by speaker
  void Compress(std::vector<float> *embeds, std::vector<float> *probs) const;

  std::vector<float> GetScores(const std::vector<float> &probs,
                               int32_t num_frames) const;

 private:
  SortformerSpeakerCacheConfig config_;

  int32_t min_positive_scores_ = 0;
  int32_t num_strong_boosted_frames_ = 0;
  int32_t num_weak_boosted_frames_ = 0;

  // (num_cache_frames_, hidden_size)
  std::vector<float> cache_embeds_;

  // (num_cache_frames_, num_speakers). Valid only if is_compressed_ is true.
  std::vector<float> cache_probs_;

  // (num_fifo_frames_, hidden_size)
  std::vector<float> fifo_;

  int32_t num_cache_frames_ = 0;
  int32_t num_fifo_frames_ = 0;
  bool is_compressed_ = false;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SORTFORMER_SPEAKER_CACHE_H_
