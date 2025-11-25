// sherpa-onnx/csrc/wave-reader-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/wave-reader.h"

#include <cstdio>
#include <fstream>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "gtest/gtest.h"

namespace sherpa_onnx {

// RAII helper class for managing temporary test files
class TempFile {
 public:
  TempFile() : TempFile("") {}

  explicit TempFile(const std::string& suffix) {
#if defined(_WIN32)
    char temp_path[MAX_PATH];
    char temp_file[MAX_PATH];
    GetTempPathA(MAX_PATH, temp_path);
    GetTempFileNameA(temp_path, "sot", 0, temp_file);
    path_ = temp_file;
    if (!suffix.empty()) {
      path_ += suffix;
      std::remove(temp_file);  // Remove the file without suffix
    }
#else
    char temp_template[] = "/tmp/sherpa_onnx_test_XXXXXX";
    int fd = mkstemp(temp_template);
    if (fd != -1) {
      close(fd);
      path_ = temp_template;
      if (!suffix.empty()) {
        path_ += suffix;
        std::remove(temp_template);  // Remove the file without suffix
      }
    }
#endif
  }

  ~TempFile() {
    if (!path_.empty()) {
      std::remove(path_.c_str());
    }
  }

  const char* path() const { return path_.c_str(); }

 private:
  std::string path_;
};

TEST(WaveReader, TestNonWavFile) {
  // Create a temporary file with non-WAV content (e.g., webm-like header)
  TempFile temp_file(".webm");

  {
    std::ofstream out(temp_file.path(), std::ios::binary);
    // Write some content that doesn't start with RIFF
    // (webm files typically start with EBML header: 0x1a45dfa3)
    const unsigned char webm_header[] = {
        0x1a, 0x45, 0xdf, 0xa3,  // EBML header signature (NOT RIFF)
        0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x1f,
        0x42, 0x86, 0x81, 0x01,
        // Add some more bytes to make it look like a real file
        0x42, 0xf7, 0x81, 0x01,
        0x42, 0xf2, 0x81, 0x04,
        'w', 'e', 'b', 'm'
    };
    out.write(reinterpret_cast<const char*>(webm_header), sizeof(webm_header));
  }

  // Test C++ API - should not segfault
  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples = ReadWave(temp_file.path(), &sample_rate, &is_ok);

  EXPECT_FALSE(is_ok);
  EXPECT_TRUE(samples.empty());
  EXPECT_EQ(sample_rate, -1);
}

TEST(WaveReader, TestNonExistentFile) {
  // Generate a unique path but don't create the file
  TempFile temp_file(".wav");

  // Test C++ API - should not segfault
  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples = ReadWave(temp_file.path(), &sample_rate, &is_ok);

  EXPECT_FALSE(is_ok);
  EXPECT_TRUE(samples.empty());
  EXPECT_EQ(sample_rate, -1);
}

TEST(WaveReader, TestTruncatedWaveFile) {
  // Create a temporary file with truncated WAV header
  TempFile temp_file(".wav");

  {
    std::ofstream out(temp_file.path(), std::ios::binary);
    // Write only partial WAV header (less than 44 bytes required)
    const unsigned char partial_wav[] = {
        'R', 'I', 'F', 'F',  // chunk_id
        0x00, 0x00, 0x00, 0x00,  // chunk_size
        'W', 'A', 'V', 'E'  // format
        // Missing the rest of the header
    };
    out.write(reinterpret_cast<const char*>(partial_wav), sizeof(partial_wav));
  }

  // Test C++ API - should not segfault
  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples = ReadWave(temp_file.path(), &sample_rate, &is_ok);

  EXPECT_FALSE(is_ok);
  EXPECT_TRUE(samples.empty());
  EXPECT_EQ(sample_rate, -1);
}

}  // namespace sherpa_onnx
