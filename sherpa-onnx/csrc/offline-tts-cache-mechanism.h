// sherpa-onnx/csrc/offline-tts-cache-mechanism.h
//
// Copyright (c)  2025  @mah92 From Iranian people to the community with love

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_CACHE_MECHANISM_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_CACHE_MECHANISM_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>  // NOLINT
#include <cstddef>  // for std::size_t

#include "sherpa-onnx/csrc/offline-tts-cache-mechanism-config.h"

namespace sherpa_onnx {

class OfflineTtsCacheMechanism {
 public:
  explicit OfflineTtsCacheMechanism(const OfflineTtsCacheMechanismConfig &config);
  ~OfflineTtsCacheMechanism();

  // Add a new wav file to the cache
  void AddWavFile(
    const std::size_t &text_hash,
    const std::vector<float> &samples,
    const int32_t sample_rate);

  // Get the cached wav file if it exists
  std::vector<float> GetWavFile(
    const std::size_t &text_hash,
    int32_t *sample_rate);

  // Get the current cache size in bytes
  int32_t GetCacheSize() const;

  // Set the cache size in bytes
  void SetCacheSize(int32_t cache_size);

  // Remove all the wav files in the cache
  void ClearCache();

  // To get total used cache size(for wav files) in bytes
  int32_t GetTotalUsedCacheSize() const;

 private:
  // Load the repeat count file
  void LoadRepeatCounts();

  // Save the repeat count file
  void SaveRepeatCounts();

  // Remove a wav file from the cache
  void RemoveWavFile(const std::size_t &text_hash);

  // Update the cache vector with the actual files in the cache folder
  void UpdateCacheVector();

  // Reduce used cache size if needed
  void EnsureCacheLimit();

  // Get the least repeated file in the cache
  std::size_t GetLeastRepeatedFile();

  // Data directory where the cache folder is located
  std::string cache_dir_;

  // Maximum number of bytes in the cache
  int32_t cache_size_bytes_;

  // Total used cache size for wav files in bytes
  int32_t used_cache_size_bytes_;

  // Map of text hash to repeat count
  std::unordered_map<std::size_t, std::size_t> repeat_counts_;

  // Vector of cached file names
  std::vector<std::size_t> cache_vector_;

  // Mutex for thread safety (recursive to avoid deadlocks)
  mutable std::recursive_mutex mutex_;

  // Time of last save
  std::chrono::steady_clock::time_point last_save_time_;

  // if cache mechanism is inited successfully
  bool cache_mechanism_inited_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CACHE_MECHANISM_H_
