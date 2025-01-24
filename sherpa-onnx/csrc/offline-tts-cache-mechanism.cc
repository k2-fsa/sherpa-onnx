// sherpa-onnx/csrc/offline-tts-cache-mechanism.cc
//
// Copyright (c)  2025  @mah92 From Iranian people to the community with love

#include "sherpa-onnx/csrc/offline-tts-cache-mechanism.h"

#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"

// Platform-specific time functions
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

namespace sherpa_onnx {

// Helper function to get the current time in seconds
static int64_t GetCurrentTimeInSeconds() {
#if defined(_WIN32)
  // Windows implementation
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  uint64_t time = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
  return static_cast<int64_t>(time / 10000000ULL - 11644473600ULL);
#else
  // Unix implementation
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec);
#endif
}

OfflineTtsCacheMechanism::OfflineTtsCacheMechanism(const std::string &cache_dir,
    int32_t cache_size)
    : cache_dir_(cache_dir),
      cache_size_bytes_(cache_size),
      used_cache_size_bytes_(0) {

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

  // Load the repeat counts
  LoadRepeatCounts();

  // Update the cache vector and calculate the total cache size
  UpdateCacheVector();

  // Initialize the last save time
  last_save_time_ = GetCurrentTimeInSeconds();

  // Indicate that initialization has been successful
  cache_mechanism_inited_ = true;
}

OfflineTtsCacheMechanism::~OfflineTtsCacheMechanism() {
  if (cache_mechanism_inited_ == false) return;

  // Save the repeat counts on destruction
  SaveRepeatCounts();
}

void OfflineTtsCacheMechanism::AddWavFile(
  const std::string &text_hash,
  const std::vector<float> &samples,
  const int32_t sample_rate) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (cache_mechanism_inited_ == false) return;

  std::string file_path = cache_dir_ + "/" + text_hash + ".wav";

  // Check if the file physically exists in the cache directory
  bool file_exists = std::filesystem::exists(file_path);

  if (!file_exists) {  // If the file does not exist, add it to the cache
    // Ensure the cache does not exceed its size limit
    EnsureCacheLimit();

    // Write the audio samples to a WAV file
    bool success = WriteWave(file_path,
       sample_rate, samples.data(), samples.size());
    if (success) {
      // Calculate size of the new WAV file and add it to the total cache size
      std::ifstream file(file_path, std::ios::binary | std::ios::ate);
      if (file.is_open()) {
        used_cache_size_bytes_ += file.tellg();
      }
    } else {
      SHERPA_ONNX_LOGE("Failed to write wav file: %s", file_path.c_str());
    }
  }
}

std::vector<float> OfflineTtsCacheMechanism::GetWavFile(
  const std::string &text_hash,
  int32_t *sample_rate) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  std::vector<float> samples;

  if (cache_mechanism_inited_ == false) return samples;

  std::string file_path = cache_dir_ + "/" + text_hash + ".wav";

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

  // Save the repeat counts every 10 minutes
  int64_t now = GetCurrentTimeInSeconds();
  if (now - last_save_time_ >= 10 * 60) {
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

  EnsureCacheLimit();
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

  // Remove repeat counts also in the repeat_counts.txt
  SaveRepeatCounts();
}

int32_t OfflineTtsCacheMechanism::GetTotalUsedCacheSize() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (cache_mechanism_inited_ == false) return 0;

  return used_cache_size_bytes_;
}

// Private functions ///////////////////////////////////////////////////

void OfflineTtsCacheMechanism::LoadRepeatCounts() {
  std::string repeat_count_file = cache_dir_ + "/repeat_counts.txt";

  // Check if the file exists
  if (!std::filesystem::exists(repeat_count_file)) {
    return;  // Skip loading if the file doesn't exist
  }

  // Open the file for reading
  std::ifstream ifs(repeat_count_file);
  if (!ifs.is_open()) {
    SHERPA_ONNX_LOGE("Failed to open repeat count file: %s",
      repeat_count_file.c_str());
    return;  // Skip loading if the file cannot be opened
  }

  // Read the file line by line
  std::string line;
  while (std::getline(ifs, line)) {
    size_t pos = line.find(' ');
    if (pos != std::string::npos) {
      std::string text_hash = line.substr(0, pos);
      int32_t count = std::stoi(line.substr(pos + 1));
      repeat_counts_[text_hash] = count;
    }
  }
}

void OfflineTtsCacheMechanism::SaveRepeatCounts() {
  std::string repeat_count_file = cache_dir_ + "/repeat_counts.txt";

  // Open the file for writing
  std::ofstream ofs(repeat_count_file);
  if (!ofs.is_open()) {
    SHERPA_ONNX_LOGE("Failed to open repeat count file for writing: %s",
      repeat_count_file.c_str());
    return;  // Skip saving if the file cannot be opened
  }

  // Write the repeat counts to the file
  for (const auto &entry : repeat_counts_) {
    ofs << entry.first << " " << entry.second;
    if (!ofs) {
      SHERPA_ONNX_LOGE("Failed to write repeat count for text hash: %s",
        entry.first.c_str());
      return;  // Stop writing if an error occurs
    }
    ofs << std::endl;
  }
}

void OfflineTtsCacheMechanism::RemoveWavFile(const std::string &text_hash) {
  std::string file_path = cache_dir_ + "/" + text_hash + ".wav";
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
      std::string text_hash = entry.path().stem().string();
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
  if (used_cache_size_bytes_ > cache_size_bytes_) {
    auto target_cache_size
      = std::max(static_cast<int> (cache_size_bytes_*0.95), 0);
    while (used_cache_size_bytes_> 0
      && used_cache_size_bytes_ > target_cache_size) {
        // Cache is full, remove the least repeated file
        std::string least_repeated_file = GetLeastRepeatedFile();
        RemoveWavFile(least_repeated_file);
    }
  }
}

std::string OfflineTtsCacheMechanism::GetLeastRepeatedFile() {
  std::string least_repeated_file;
  int32_t min_count = std::numeric_limits<int32_t>::max();

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
