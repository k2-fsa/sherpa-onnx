// sherpa-onnx/csrc/wave-reader-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/wave-reader.h"

#include <fstream>
#include <string>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(WaveReader, TestNonWavFile) {
  // Create a temporary file with non-WAV content (e.g., webm-like header)
  const char* temp_file = "/tmp/test_non_wav_file.webm";

  {
    std::ofstream out(temp_file, std::ios::binary);
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
  std::vector<float> samples = ReadWave(temp_file, &sample_rate, &is_ok);

  EXPECT_FALSE(is_ok);
  EXPECT_TRUE(samples.empty());
  EXPECT_EQ(sample_rate, -1);

  // Clean up
  std::remove(temp_file);
}

TEST(WaveReader, TestNonExistentFile) {
  const char* non_existent = "/tmp/this_file_does_not_exist_12345.wav";

  // Test C++ API - should not segfault
  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples = ReadWave(non_existent, &sample_rate, &is_ok);

  EXPECT_FALSE(is_ok);
  EXPECT_TRUE(samples.empty());
}

TEST(WaveReader, TestTruncatedWaveFile) {
  // Create a temporary file with truncated WAV header
  const char* temp_file = "/tmp/test_truncated_wave.wav";

  {
    std::ofstream out(temp_file, std::ios::binary);
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
  std::vector<float> samples = ReadWave(temp_file, &sample_rate, &is_ok);

  EXPECT_FALSE(is_ok);
  EXPECT_TRUE(samples.empty());

  // Clean up
  std::remove(temp_file);
}

}  // namespace sherpa_onnx
