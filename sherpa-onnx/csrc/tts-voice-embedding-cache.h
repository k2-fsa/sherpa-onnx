// sherpa-onnx/csrc/tts-voice-embedding-cache.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Shared thread-safe LRU cache for voice embeddings, used by multiple TTS
// implementations (Pocket TTS, Pocket ZhEn TTS, etc.).

#ifndef SHERPA_ONNX_CSRC_TTS_VOICE_EMBEDDING_CACHE_H_
#define SHERPA_ONNX_CSRC_TTS_VOICE_EMBEDDING_CACHE_H_

#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

// Compute a hash over a float array, used as cache key for voice embeddings.
inline size_t ComputeVoiceEmbeddingHash(const float *p, size_t n) {
  size_t hash = 0;

  auto hash_combine = [](size_t &seed, size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
  };

  hash_combine(hash, n);

  for (size_t i = 0; i < n; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &p[i], sizeof(float));
    hash_combine(hash, bits);
  }

  return hash;
}

// Thread-safe LRU cache for voice embeddings.
// Key is a hash of the reference audio; value is (data, shape).
class VoiceEmbeddingCache {
 public:
  using Embedding = std::pair<std::vector<float>, std::vector<int64_t>>;
  using EmbeddingPtr = std::shared_ptr<Embedding>;

  static constexpr size_t kDefaultCapacity = 50;

  explicit VoiceEmbeddingCache(size_t cap = kDefaultCapacity)
      : capacity_(cap) {}

  EmbeddingPtr Get(size_t key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = map_.find(key);
    if (it == map_.end()) {
      return nullptr;  // cache miss
    }

    // Move to front (most recently used)
    if (it->second != lru_list_.begin()) {
      lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
    }

    return it->second->second;  // copy shared_ptr
  }

  void Put(size_t key, std::vector<float> data, std::vector<int64_t> shape) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (capacity_ == 0) {
      return;
    }

    auto it = map_.find(key);

    // If exists, update and move to front
    if (it != map_.end()) {
      it->second->second =
          std::make_shared<Embedding>(std::move(data), std::move(shape));

      if (it->second != lru_list_.begin()) {
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
      }
      return;
    }

    // Evict if full
    if (lru_list_.size() >= capacity_) {
      auto &last = lru_list_.back();
      size_t last_key = last.first;

      map_.erase(last_key);
      lru_list_.pop_back();  // shared_ptr released here
    }

    // Insert new at front
    lru_list_.emplace_front(
        key, std::make_shared<Embedding>(std::move(data), std::move(shape)));

    map_[key] = lru_list_.begin();
  }

  void SetCapacity(int32_t cap) {
    if (cap < 0) {
      SHERPA_ONNX_LOGE("voice_embedding_cache_capacity must be >= 0. Given: %d",
                       cap);
      SHERPA_ONNX_EXIT(-1);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = cap;

    while (lru_list_.size() > capacity_) {
      auto &last = lru_list_.back();
      size_t last_key = last.first;

      map_.erase(last_key);
      lru_list_.pop_back();
    }
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_list_.size();
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    map_.clear();
    lru_list_.clear();
  }

 private:
  using ListNode = std::pair<size_t, EmbeddingPtr>;
  using ListIt = std::list<ListNode>::iterator;

  mutable std::mutex mutex_;
  size_t capacity_;

  // Front = most recently used
  std::list<ListNode> lru_list_;

  // Key -> iterator into lru_list_
  std::unordered_map<size_t, ListIt> map_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TTS_VOICE_EMBEDDING_CACHE_H_
