// sherpa-onnx/csrc/sortformer-speaker-cache-test.cc
//
// Copyright (c)  2026  Silvio Tomatis

#include "sherpa-onnx/csrc/sortformer-speaker-cache.h"

#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

namespace {

constexpr float kSilence = -1;

// One-dimensional embeddings hold the frame index, so that we can track
// which frames are kept.
SortformerSpeakerCacheConfig SmallConfig() {
  SortformerSpeakerCacheConfig config;
  config.num_speakers = 2;
  config.hidden_size = 1;
  config.subsampling_factor = 1;
  config.fifo_length = 0;
  config.speaker_cache_update_period = 1;
  config.speaker_cache_length = 4;
  config.num_silence_frames = 1;
  config.silence_embeds = {kSilence};
  return config;
}

// Push one frame. probs[i] holds the probabilities of frame i. The step
// probabilities are those of the cached frames, followed by the new frame.
void Push(SortformerSpeakerCache *cache, int32_t frame,
          const std::vector<std::vector<float>> &probs) {
  std::vector<float> step_probs;
  for (float f : cache->GetEmbeds()) {
    const auto &p = f == kSilence ? std::vector<float>{0, 0} : probs[f];
    step_probs.insert(step_probs.end(), p.begin(), p.end());
  }
  step_probs.insert(step_probs.end(), probs[frame].begin(), probs[frame].end());

  float embed = static_cast<float>(frame);
  cache->Update(&embed, 1, step_probs.data(), cache->NumCachedFrames() + 1);
}

}  // namespace

TEST(SortformerSpeakerCache, FifoOverflow) {
  SortformerSpeakerCacheConfig config = SmallConfig();
  config.fifo_length = 4;
  config.speaker_cache_update_period = 3;
  config.speaker_cache_length = 100;

  SortformerSpeakerCache cache(config);

  std::vector<float> embeds = {0, 1, 2, 3};
  std::vector<float> probs(4 * 2, 0.9f);
  cache.Update(embeds.data(), 4, probs.data(), 4);

  // The FIFO is full but not overflowing
  EXPECT_EQ(cache.NumSpeakerCacheFrames(), 0);
  EXPECT_EQ(cache.NumFifoFrames(), 4);

  embeds = {4, 5};
  probs.assign(6 * 2, 0.9f);
  cache.Update(embeds.data(), 2, probs.data(), 6);

  // 6 frames overflow a FIFO of 4. At least speaker_cache_update_period
  // frames are moved to the speaker cache.
  EXPECT_EQ(cache.NumSpeakerCacheFrames(), 3);
  EXPECT_EQ(cache.NumFifoFrames(), 3);
  EXPECT_EQ(cache.GetEmbeds(), (std::vector<float>{0, 1, 2, 3, 4, 5}));

  embeds = {6, 7, 8, 9, 10, 11, 12};
  probs.assign(13 * 2, 0.9f);
  cache.Update(embeds.data(), 7, probs.data(), 13);

  // 10 frames in the FIFO: 6 frames are moved to keep 4
  EXPECT_EQ(cache.NumSpeakerCacheFrames(), 9);
  EXPECT_EQ(cache.NumFifoFrames(), 4);
}

TEST(SortformerSpeakerCache, CompressGroupsBySpeaker) {
  // Frames 0 and 1 are speaker 0, frames 2 and 3 are speaker 1 and frame 4
  // is silence.
  std::vector<std::vector<float>> probs = {
      {0.9f, 0.1f}, {0.9f, 0.1f}, {0.1f, 0.9f}, {0.1f, 0.9f}, {0.1f, 0.1f},
  };

  SortformerSpeakerCache cache(SmallConfig());
  for (int32_t i = 0; i != 4; ++i) {
    Push(&cache, i, probs);
  }
  EXPECT_EQ(cache.GetEmbeds(), (std::vector<float>{0, 1, 2, 3}));

  // The 5th frame overflows the speaker cache of 4 frames.
  //
  // Every speaker has a silence slot. Each one keeps its best frame:
  // its first frame, as ties go to the lower index.
  Push(&cache, 4, probs);
  EXPECT_EQ(cache.NumSpeakerCacheFrames(), 4);
  EXPECT_EQ(cache.NumFifoFrames(), 0);
  EXPECT_EQ(cache.GetEmbeds(), (std::vector<float>{0, kSilence, 2, kSilence}));
}

TEST(SortformerSpeakerCache, SilenceOnly) {
  std::vector<std::vector<float>> probs(6, {0.1f, 0.2f});

  SortformerSpeakerCache cache(SmallConfig());
  for (int32_t i = 0; i != 6; ++i) {
    Push(&cache, i, probs);
  }

  // There is no speech to keep: all slots hold the silence embedding
  EXPECT_EQ(cache.GetEmbeds(), std::vector<float>(4, kSilence));
}

}  // namespace sherpa_onnx
