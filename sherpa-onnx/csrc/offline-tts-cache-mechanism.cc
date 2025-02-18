// sherpa-onnx/csrc/offline-tts-cache-mechanism.cc
//
// Copyright (c)  2025  @mah92 From Iranian people to the community with love

#include "sherpa-onnx/csrc/offline-tts-cache-mechanism.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <thread>
#include <cstddef>  // for std::size_t

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"

namespace sherpa_onnx {

OfflineTtsCacheMechanism::OfflineTtsCacheMechanism(
  const OfflineTtsCacheMechanismConfig &config)
  : cache_dir_(config.cache_dir),
      cache_size_bytes_(config.cache_size),
      used_cache_size_bytes_(0)
{
  // Create the cache directory if it doesn't exist
  if (!std::filesystem::exists(cache_dir_)) {
    bool dir_created = std::filesystem::create_directory(cache_dir_);
    if (!dir_created) {
      SHERPA_ONNX_LOGE("Unable to create cache directory: %s",
        cache_dir_.c_str());
      SHERPA_ONNX_LOGE("Cache mechanism disabled!");
      cache_mechanism_inited_ = false;
      return;
    }
  }

  if(cache_size_bytes_ == -1) 
    cache_size_bytes_ = INT32_MAX;  // Unlimited cache size

  // Load the repeat counts
  LoadRepeatCounts();

  // Update the cache vector and calculate the total cache size
  UpdateCacheVector();

  // Initialize the last save time
  last_save_time_ = std::chrono::steady_clock::now();

  // Indicate that initialization has been successful
  cache_mechanism_inited_ = true;
}

OfflineTtsCacheMechanism::~OfflineTtsCacheMechanism() {
  if (cache_mechanism_inited_ == false) return;

  // Save the repeat counts on destruction
  SaveRepeatCounts();
}

void OfflineTtsCacheMechanism::AddWavFile(
  const std::size_t &text_hash,
  const std::vector<float> &samples,
  const int32_t sample_rate) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (cache_mechanism_inited_ == false) return;

  std::string file_path = cache_dir_ + "/" + std::to_string(text_hash) + ".wav";

  // Check if the file physically exists in the cache directory
  bool file_exists = std::filesystem::exists(file_path);

  if (!file_exists) {  // If the file does not exist, add it to the cache

    // Write the audio samples to a WAV file
    bool success = WriteWave(file_path,
       sample_rate, samples.data(), samples.size());
    if (success) {
      // Calculate size of the new WAV file and add it to the total cache size
      std::ifstream file(file_path, std::ios::binary | std::ios::ate);
      if (file.is_open()) {
        used_cache_size_bytes_ += file.tellg();
      }

      // Ensure the cache does not exceed its size limit, non-blocking
      EnsureCacheLimit();

    } else {
      SHERPA_ONNX_LOGE("Failed to write wav file: %s", file_path.c_str());
    }
  }
}

std::vector<float> OfflineTtsCacheMechanism::GetWavFile(
  const std::size_t &text_hash,
  int32_t *sample_rate) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  std::vector<float> samples;

  if (cache_mechanism_inited_ == false) return samples;

  std::string file_path = cache_dir_ + "/" + std::to_string(text_hash) + ".wav";

  if (std::filesystem::exists(file_path)) {
    bool is_ok = false;
    samples = ReadWave(file_path, sample_rate, &is_ok);

    if (is_ok == false) {
      SHERPA_ONNX_LOGE("Failed to read cached file: %s", file_path.c_str());
    }
  }

  // Ensure the text_hash exists in the map before incrementing the count
  if (repeat_counts_.find(text_hash) == repeat_counts_.end()) {
    repeat_counts_[text_hash] = 1;  // Initialize if it doesn't exist
  } else {
    repeat_counts_[text_hash]++;  // Increment the repeat count
  }

  // Save the repeat counts every minute
  auto now = std::chrono::steady_clock::now();
  if (std::chrono::duration_cast<std::chrono::seconds>(
    now - last_save_time_).count() >= 1 * 60) {
    SaveRepeatCounts();
    last_save_time_ = now;
  }

  return samples;
}

int32_t OfflineTtsCacheMechanism::GetCacheSize() const {
  if (cache_mechanism_inited_ == false) return 0;

  return cache_size_bytes_;
}

void OfflineTtsCacheMechanism::SetCacheSize(int32_t cache_size) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (cache_mechanism_inited_ == false) return;

  cache_size_bytes_ = cache_size;

  if(cache_size == 0) {
    ClearCache();
  } else {
    EnsureCacheLimit();
  }
}

void OfflineTtsCacheMechanism::ClearCache() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (cache_mechanism_inited_ == false) return;

  // Remove all WAV files in the cache directory
  for (const auto &entry : std::filesystem::directory_iterator(cache_dir_)) {
    if (entry.path().extension() == ".wav") {
      std::filesystem::remove(entry.path());
    }
  }

  // Reset the total cache size to 0
  used_cache_size_bytes_ = 0;

  // Clear the repeat counts and cache vector
  repeat_counts_.clear();
  cache_vector_.clear();

  // Remove repeat counts also in the repeat_counts file
  SaveRepeatCounts();
}

int32_t OfflineTtsCacheMechanism::GetTotalUsedCacheSize() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (cache_mechanism_inited_ == false) return 0;

  return used_cache_size_bytes_;
}

// Private functions ///////////////////////////////////////////////////

void OfflineTtsCacheMechanism::LoadRepeatCounts() {
  std::string repeat_count_file = cache_dir_ + "/repeat_counts.bin";

  // Check if the file exists
  if (!std::filesystem::exists(repeat_count_file)) {
    return;  // Skip loading if the file doesn't exist
  }

  // Open the file for reading in binary mode
  std::ifstream ifs(repeat_count_file, std::ios::binary);
  if (!ifs.is_open()) {
    SHERPA_ONNX_LOGE("Failed to open repeat count file: %s",
      repeat_count_file.c_str());
    return;  // Skip loading if the file cannot be opened
  }

  // Read the number of entries
  size_t num_entries;
  ifs.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));

  // Read each entry
  for (size_t i = 0; i < num_entries; ++i) {
    std::size_t text_hash;
    std::size_t count;
    ifs.read(reinterpret_cast<char*>(&text_hash), sizeof(text_hash));
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    repeat_counts_[text_hash] = count;
  }
}

void OfflineTtsCacheMechanism::SaveRepeatCounts() {
  // Start timing
  auto start_time = std::chrono::steady_clock::now();

  std::string repeat_count_file = cache_dir_ + "/repeat_counts.bin";

  // Open the file for writing in binary mode
  std::ofstream ofs(repeat_count_file, std::ios::binary);
  if (!ofs.is_open()) {
    SHERPA_ONNX_LOGE("Failed to open repeat count file for writing: %s",
      repeat_count_file.c_str());
    return;  // Skip saving if the file cannot be opened
  }

  // Write the number of entries
  size_t num_entries = repeat_counts_.size();
  ofs.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));

  // Write each entry
  for (const auto &entry : repeat_counts_) {
    ofs.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
    ofs.write(reinterpret_cast<const char*>(&entry.second), sizeof(entry.second));
  }

  // End timing
  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  // Print the time taken
  SHERPA_ONNX_LOGE("SaveRepeatCounts took %lld milliseconds", elapsed_time);
}

void OfflineTtsCacheMechanism::RemoveWavFile(const std::size_t &text_hash) {
  std::string file_path = cache_dir_ + "/" 
                            + std::to_string(text_hash) + ".wav";
  if (std::filesystem::exists(file_path)) {
    // Subtract the size of the removed WAV file from the total cache size
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
      used_cache_size_bytes_ -= file.tellg();
      file.close();
    }
    std::filesystem::remove(file_path);
  }

  // Remove the entry from the repeat counts and cache vector
  if (repeat_counts_.find(text_hash) != repeat_counts_.end()) {
    repeat_counts_.erase(text_hash);
    cache_vector_.erase(
      std::remove(cache_vector_.begin(), cache_vector_.end(), text_hash),
      cache_vector_.end());
  }
}

void OfflineTtsCacheMechanism::UpdateCacheVector() {
  used_cache_size_bytes_ = 0;  // Reset total cache size before recalculating

  for (const auto &entry : std::filesystem::directory_iterator(cache_dir_)) {
    if (entry.path().extension() == ".wav") {
      std::string text_hash_str = entry.path().stem().string();
      std::size_t text_hash = std::stoull(text_hash_str);
      if (repeat_counts_.find(text_hash) == repeat_counts_.end()) {
        // Remove the file if it's not in the repeat count file (orphaned file)
        std::filesystem::remove(entry.path());
      } else {
        // Add the size of the WAV file to the total cache size
        std::ifstream file(entry.path(), std::ios::binary | std::ios::ate);
        if (file.is_open()) {
          used_cache_size_bytes_ += file.tellg();
        }
        cache_vector_.push_back(text_hash);
      }
    }
  }
}

void OfflineTtsCacheMechanism::EnsureCacheLimit() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);  // Lock the mutex for the entire function

  if (used_cache_size_bytes_ > cache_size_bytes_) {
    // Launch a new thread to handle cache cleanup in a non-blocking way
    std::thread([this]() {
      std::lock_guard<std::recursive_mutex> lock(mutex_);  // Lock the mutex for the cleanup process

      auto target_cache_size = std::max(static_cast<int>(cache_size_bytes_ * 0.95), 0);
      while (used_cache_size_bytes_ > 0
             && used_cache_size_bytes_ > target_cache_size) {
        // Cache is full, remove the least repeated file
        std::size_t least_repeated_file = GetLeastRepeatedFile();
        RemoveWavFile(least_repeated_file);
      }
    }).detach();  // Detach the thread to run independently
  }
}

std::size_t OfflineTtsCacheMechanism::GetLeastRepeatedFile() {
  std::size_t least_repeated_file = 0;
  std::size_t min_count = std::numeric_limits<std::size_t>::max();

  for (const auto &entry : repeat_counts_) {
    if (entry.second <= 1) {
      least_repeated_file = entry.first;
      return least_repeated_file;
    }

    if (entry.second < min_count) {
      min_count = entry.second;
      least_repeated_file = entry.first;
    }
  }

  return least_repeated_file;
}

}  // namespace sherpa_onnx
