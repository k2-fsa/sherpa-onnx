// sherpa-onnx/csrc/offline-tts-cache-mechanism.cc
//
// @mah92 From Iranian people to the comunity with love

#include "sherpa-onnx/csrc/offline-tts-cache-mechanism.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <algorithm>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"

namespace sherpa_onnx {

CacheMechanism::CacheMechanism(const std::string &cache_dir, int32_t cache_size)
    : cache_dir_(cache_dir), cache_size_bytes_(cache_size), used_cache_size_bytes_(0) {
  // Create the cache directory if it doesn't exist
  if (!std::filesystem::exists(cache_dir_)) {
    std::filesystem::create_directory(cache_dir_);
    // SHERPA_ONNX_LOGE("Created cache directory: %s", cache_dir_.c_str());
  }

  // Load the repeat counts
  LoadRepeatCounts();

  // Update the cache vector and calculate the total cache size
  UpdateCacheVector();

  // Initialize the last save time
  last_save_time_ = std::chrono::steady_clock::now();
}

CacheMechanism::~CacheMechanism() {
  // Save the repeat counts on destruction
  SaveRepeatCounts();
}

void CacheMechanism::AddWavFile(const std::string &text_hash, const std::vector<float> &samples, int32_t sample_rate) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  std::string file_path = cache_dir_ + "/" + text_hash + ".wav";

  // Check if the file physically exists in the cache directory
  bool file_exists = std::filesystem::exists(file_path);

  if (!file_exists) { // If the file does not exist, add it to the cache
    // Ensure the cache does not exceed its size limit
    EnsureCacheLimit();

    // Write the audio samples to a WAV file
    bool success = WriteWave(file_path, sample_rate, samples.data(), samples.size());
    if (success) {
      // Calculate the size of the new WAV file and add it to the total cache size
      std::ifstream file(file_path, std::ios::binary | std::ios::ate);
      if (file.is_open()) {
        used_cache_size_bytes_ += file.tellg();
        file.close();
      }
      // SHERPA_ONNX_LOGE("Added new wav file to cache: %s, samples:%d", file_path.c_str(), samples.size());
    } else {
      SHERPA_ONNX_LOGE("Failed to write wav file: %s", file_path.c_str());
    }
  }

  // Ensure the text_hash exists in the map before incrementing the count
  if (repeat_counts_.find(text_hash) == repeat_counts_.end()) {
    repeat_counts_[text_hash] = 0; // Initialize if it doesn't exist
    cache_vector_.push_back(text_hash); // Add the text_hash to the cache vector
  }
  repeat_counts_[text_hash]++; // Increment the repeat count

  // Save the repeat counts every 10 minutes
  auto now = std::chrono::steady_clock::now();
  if (std::chrono::duration_cast<std::chrono::seconds>(now - last_save_time_).count() >= 10 * 60) {
    SaveRepeatCounts();
    last_save_time_ = now;
  }
}

std::vector<float> CacheMechanism::GetWavFile(const std::string &text_hash, int32_t &sample_rate) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  std::vector<float> samples;
  std::string file_path = cache_dir_ + "/" + text_hash + ".wav";

  if (std::filesystem::exists(file_path)) {
    bool is_ok = false;
    samples = ReadWave(file_path, &sample_rate, &is_ok);

    if (is_ok) {
      // SHERPA_ONNX_LOGE("Retrieved cached audio for text hash: %s, sample count: %d, freq:%dHz", text_hash.c_str(), samples.size(), sample_rate);
    } else {
      SHERPA_ONNX_LOGE("Failed to read cached file: %s", file_path.c_str());
    }
  } else {
    // SHERPA_ONNX_LOGE("Cached audio not found for text hash: %s", text_hash.c_str());
  }

  // Ensure the text_hash exists in the map before incrementing the count
  if (repeat_counts_.find(text_hash) == repeat_counts_.end()) {
    repeat_counts_[text_hash] = 0; // Initialize if it doesn't exist
  }
  repeat_counts_[text_hash]++; // Increment the repeat count

  return samples;
}

int32_t CacheMechanism::GetCacheSize() const {
  return cache_size_bytes_;
}

void CacheMechanism::SetCacheSize(int32_t cache_size) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  cache_size_bytes_ = cache_size;

  EnsureCacheLimit();
}

void CacheMechanism::ClearCache() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Remove all WAV files in the cache directory
  for (const auto &entry : std::filesystem::directory_iterator(cache_dir_)) {
    if (entry.path().extension() == ".wav") {
      std::filesystem::remove(entry.path());
      // SHERPA_ONNX_LOGE("Removed wav file: %s", entry.path().c_str());
    }
  }

  // SHERPA_ONNX_LOGE("Removed all wav files!");

  // Reset the total cache size to 0
  used_cache_size_bytes_ = 0;

  // SHERPA_ONNX_LOGE("Cache cleared successfully.");
}

int64_t CacheMechanism::GetTotalUsedCacheSize() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return used_cache_size_bytes_;
}

void CacheMechanism::LoadRepeatCounts() {
  std::string repeat_count_file = cache_dir_ + "/repeat_counts.txt";

  // Check if the file exists
  if (!std::filesystem::exists(repeat_count_file)) {
    // SHERPA_ONNX_LOGE("Repeat count file does not exist. Starting with an empty cache.");
    return;  // Skip loading if the file doesn't exist
  }

  // Open the file for reading
  std::ifstream ifs(repeat_count_file);
  if (!ifs.is_open()) {
    SHERPA_ONNX_LOGE("Failed to open repeat count file: %s", repeat_count_file.c_str());
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
      // SHERPA_ONNX_LOGE("Loaded repeat count for text hash: %s, count: %d", text_hash.c_str(), count);
    }
  }
}

void CacheMechanism::SaveRepeatCounts() {
  std::string repeat_count_file = cache_dir_ + "/repeat_counts.txt";

  // Open the file for writing
  std::ofstream ofs(repeat_count_file);
  if (!ofs.is_open()) {
    SHERPA_ONNX_LOGE("Failed to open repeat count file for writing: %s", repeat_count_file.c_str());
    return;  // Skip saving if the file cannot be opened
  }

  // Write the repeat counts to the file
  for (const auto &entry : repeat_counts_) {
    ofs << entry.first << " " << entry.second;
    if (!ofs) {
      SHERPA_ONNX_LOGE("Failed to write repeat count for text hash: %s", entry.first.c_str());
      return;  // Stop writing if an error occurs
    }
    ofs << std::endl;
  }
}

void CacheMechanism::RemoveWavFile(const std::string &text_hash) {
  std::string file_path = cache_dir_ + "/" + text_hash + ".wav";
  if (std::filesystem::exists(file_path)) {
    // Subtract the size of the removed WAV file from the total cache size
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
      used_cache_size_bytes_ -= file.tellg();
      file.close();
    }
    std::filesystem::remove(file_path);
    // SHERPA_ONNX_LOGE("Removed wav file: %s", file_path.c_str());
  }

  // Remove the entry from the repeat counts and cache vector
  if (repeat_counts_.find(text_hash) != repeat_counts_.end()) {
    repeat_counts_.erase(text_hash);
    cache_vector_.erase(std::remove(cache_vector_.begin(), cache_vector_.end(), text_hash), cache_vector_.end());
  }
}

void CacheMechanism::UpdateCacheVector() {
  used_cache_size_bytes_ = 0; // Reset the total cache size before recalculating

  for (const auto &entry : std::filesystem::directory_iterator(cache_dir_)) {
    if (entry.path().extension() == ".wav") {
      std::string text_hash = entry.path().stem().string();
      if (repeat_counts_.find(text_hash) == repeat_counts_.end()) {
        // Remove the file if it's not in the repeat count file
        std::filesystem::remove(entry.path());
        // SHERPA_ONNX_LOGE("Removed orphaned wav file: %s", entry.path().c_str());
      } else {
        // Add the size of the WAV file to the total cache size
        std::ifstream file(entry.path(), std::ios::binary | std::ios::ate);
        if (file.is_open()) {
          used_cache_size_bytes_ += file.tellg();
          file.close();
        }
        cache_vector_.push_back(text_hash);
      }
    }
  }
}

void CacheMechanism::EnsureCacheLimit() {
  if(used_cache_size_bytes_ > cache_size_bytes_) {
    auto target_cache_size = std::max((int)(cache_size_bytes_*0.95), 0); //Remove more to prevent deleting every step
    while (used_cache_size_bytes_> 0 && used_cache_size_bytes_ > target_cache_size) {
        // Cache is full, remove the least repeated file
        std::string least_repeated_file = GetLeastRepeatedFile();
        RemoveWavFile(least_repeated_file);
        // SHERPA_ONNX_LOGE("Cache full, removed least repeated file: %s", least_repeated_file.c_str());
    }
  }
}

std::string CacheMechanism::GetLeastRepeatedFile() {
  std::string least_repeated_file;
  int32_t min_count = std::numeric_limits<int32_t>::max();

  for (const auto &entry : repeat_counts_) {
    if (entry.second < min_count) {
      min_count = entry.second;
      least_repeated_file = entry.first;
    }
  }

  return least_repeated_file;
}

}  // namespace sherpa_onnx