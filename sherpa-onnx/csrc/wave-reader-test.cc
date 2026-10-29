// sherpa-onnx/csrc/wave-reader-test.cc
//
// Copyright (c)  2025  Posit Software, PBC

#include "sherpa-onnx/csrc/wave-reader.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/file-utils.h"

namespace sherpa_onnx {

// RAII helper class for managing temporary test files
class TempFile {
 public:
  TempFile() : TempFile("") {}

  explicit TempFile(const std::string &suffix) {
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

  const char *path() const { return path_.c_str(); }

 private:
  std::string path_;
};

TEST(WaveReader, TestNonWavFile) {
  // Create a temporary file with non-WAV content (e.g., webm-like header)
  TempFile temp_file(".webm");

  {
    auto out = OpenOutputFile(temp_file.path(), std::ios::binary);
    // Write some content that doesn't start with RIFF
    // (webm files typically start with EBML header: 0x1a45dfa3)
    const unsigned char webm_header[] = {
        0x1a, 0x45, 0xdf, 0xa3,  // EBML header signature (NOT RIFF)
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0x42, 0x86, 0x81, 0x01,
        // Add some more bytes to make it look like a real file
        0x42, 0xf7, 0x81, 0x01, 0x42, 0xf2, 0x81, 0x04, 'w', 'e', 'b', 'm'};
    out.write(reinterpret_cast<const char *>(webm_header), sizeof(webm_header));
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
    auto out = OpenOutputFile(temp_file.path(), std::ios::binary);
    // Write only partial WAV header (less than 44 bytes required)
    const unsigned char partial_wav[] = {
        'R',  'I',  'F',
        'F',  // chunk_id
        0x00, 0x00, 0x00,
        0x00,  // chunk_size
        'W',  'A',  'V',
        'E'  // format
             // Missing the rest of the header
    };
    out.write(reinterpret_cast<const char *>(partial_wav), sizeof(partial_wav));
  }

  // Test C++ API - should not segfault
  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples = ReadWave(temp_file.path(), &sample_rate, &is_ok);

  EXPECT_FALSE(is_ok);
  EXPECT_TRUE(samples.empty());
  EXPECT_EQ(sample_rate, -1);
}

namespace {

void Put16(std::string *s, uint16_t v) {
  s->push_back(static_cast<char>(v & 0xff));
  s->push_back(static_cast<char>(v >> 8));
}

void Put32(std::string *s, uint32_t v) {
  for (int32_t i = 0; i != 4; ++i) {
    s->push_back(static_cast<char>((v >> (8 * i)) & 0xff));
  }
}

std::string Chunk(const std::string &id, const std::string &data,
                  uint32_t size) {
  std::string s = id;
  Put32(&s, size);
  return s + data;
}

std::string Chunk(const std::string &id, const std::string &data) {
  return Chunk(id, data, static_cast<uint32_t>(data.size()));
}

// A RIFF/WAVE file whose "fmt " chunk is followed by `chunks`
std::string MakeWave(int16_t audio_format, int16_t num_channels,
                     int32_t sample_rate, int16_t bits_per_sample,
                     const std::string &chunks) {
  std::string fmt;
  Put16(&fmt, audio_format);
  Put16(&fmt, num_channels);
  Put32(&fmt, sample_rate);
  Put32(&fmt, sample_rate * num_channels * bits_per_sample / 8);
  Put16(&fmt, num_channels * bits_per_sample / 8);
  Put16(&fmt, bits_per_sample);

  std::string body = "WAVE" + Chunk("fmt ", fmt) + chunks;
  std::string riff = "RIFF";
  Put32(&riff, static_cast<uint32_t>(body.size()));
  return riff + body;
}

std::vector<std::vector<float>> Read(const std::string &wave, bool *is_ok) {
  std::istringstream is(wave);
  int32_t sample_rate = -1;
  return ReadWaveMultiChannel(is, &sample_rate, is_ok);
}

}  // namespace

TEST(WaveReader, TestTrailingPartialFrameIsIgnored) {
  std::string data;
  for (int32_t i = 1; i <= 5; ++i) {
    Put16(&data, static_cast<uint16_t>(i * 1024));
  }
  // 5 int16 samples in 2 channels: 2 whole frames plus half of one
  bool is_ok = false;
  auto channels = Read(MakeWave(1, 2, 16000, 16, Chunk("data", data)), &is_ok);

  ASSERT_TRUE(is_ok);
  ASSERT_EQ(channels.size(), 2);
  EXPECT_EQ(channels[0], (std::vector<float>{1024 / 32768., 3072 / 32768.}));
  EXPECT_EQ(channels[1], (std::vector<float>{2048 / 32768., 4096 / 32768.}));

  // The same with 8-bit samples: 3 bytes in 2 channels
  auto channels8 =
      Read(MakeWave(1, 2, 8000, 8, Chunk("data", "\x80\xc0\x40")), &is_ok);

  ASSERT_TRUE(is_ok);
  ASSERT_EQ(channels8.size(), 2);
  EXPECT_EQ(channels8[0], std::vector<float>{0});
  EXPECT_EQ(channels8[1], std::vector<float>{0.5});
}

TEST(WaveReader, Test32BitDataSizeNotMultipleOf4) {
  std::string int32_data;
  Put32(&int32_data, 0x40000000);
  int32_data += "\x01\x02";  // 2 stray bytes

  bool is_ok = false;
  auto int32_channels =
      Read(MakeWave(1, 1, 16000, 32, Chunk("data", int32_data)), &is_ok);
  ASSERT_TRUE(is_ok);
  EXPECT_EQ(int32_channels[0], std::vector<float>{0.5});

  float value = 0.25f;
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  std::string float_data;
  Put32(&float_data, bits);
  float_data += "\x01\x02\x03";  // 3 stray bytes

  auto float_channels =
      Read(MakeWave(3, 1, 16000, 32, Chunk("data", float_data)), &is_ok);
  ASSERT_TRUE(is_ok);
  EXPECT_EQ(float_channels[0], std::vector<float>{0.25});
}

TEST(WaveReader, TestInt32SamplesKeepTheirSign) {
  std::string data;
  Put32(&data, 0x40000000);  // +2^30
  Put32(&data, 0xc0000000);  // -2^30

  bool is_ok = false;
  auto channels = Read(MakeWave(1, 1, 16000, 32, Chunk("data", data)), &is_ok);

  ASSERT_TRUE(is_ok);
  EXPECT_EQ(channels[0], (std::vector<float>{0.5, -0.5}));
}

TEST(WaveReader, TestOddSizedChunkIsFollowedByAPadByte) {
  std::string data;
  Put16(&data, 16384);
  // RIFF pads a chunk with an odd size to an even length
  std::string list = Chunk("LIST", "abc") + std::string(1, '\0');

  bool is_ok = false;
  auto channels =
      Read(MakeWave(1, 1, 16000, 16, list + Chunk("data", data)), &is_ok);

  ASSERT_TRUE(is_ok);
  EXPECT_EQ(channels[0], std::vector<float>{0.5});
}

TEST(WaveReader, TestNegativeChunkSizeIsRejected) {
  std::string data;
  Put16(&data, 16384);

  bool is_ok = true;
  // Skipping -8 bytes would land on this chunk again, forever
  auto channels = Read(
      MakeWave(1, 1, 16000, 16,
               Chunk("junk", "", 0xfffffff8u) + Chunk("data", data)),  // -8
      &is_ok);
  EXPECT_FALSE(is_ok);
  EXPECT_TRUE(channels.empty());

  is_ok = true;
  channels =
      Read(MakeWave(1, 1, 8000, 8, Chunk("data", "", 0xfffffffeu)),  // -2
           &is_ok);
  EXPECT_FALSE(is_ok);
  EXPECT_TRUE(channels.empty());
}

}  // namespace sherpa_onnx
