// sherpa-onnx/csrc/wave-writer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/wave-writer.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {
namespace {

// see http://soundfile.sapp.org/doc/WaveFormat/
//
// Note: We assume little endian here
// TODO(fangjun): Support big endian
struct WaveHeader {
  int32_t chunk_id;
  int32_t chunk_size;
  int32_t format;
  int32_t subchunk1_id;
  int32_t subchunk1_size;
  int16_t audio_format;
  int16_t num_channels;
  int32_t sample_rate;
  int32_t byte_rate;
  int16_t block_align;
  int16_t bits_per_sample;
  int32_t subchunk2_id;    // a tag of this chunk
  int32_t subchunk2_size;  // size of subchunk2
};

}  // namespace

int64_t WaveFileSize(int32_t n_samples, int32_t num_channels /*= 1*/) {
  return sizeof(WaveHeader) + n_samples * sizeof(int16_t) * num_channels;
}

void WriteWave(char *buffer, int32_t sampling_rate, const float *samples,
               int32_t n) {
  WriteWave(buffer, sampling_rate, samples, nullptr, n);
}

bool WriteWave(const std::string &filename, int32_t sampling_rate,
               const float *samples, int32_t n) {
  return WriteWave(filename, sampling_rate, samples, nullptr, n);
}

bool WriteWave(const std::string &filename, int32_t sampling_rate,
               const float *samples_ch0, const float *samples_ch1, int32_t n) {
  std::string buffer;
  buffer.resize(WaveFileSize(n, samples_ch1 == nullptr ? 1 : 2));

  WriteWave(buffer.data(), sampling_rate, samples_ch0, samples_ch1, n);

  std::ofstream os(filename, std::ios::binary);
  if (!os) {
    SHERPA_ONNX_LOGE("Failed to create '%s'", filename.c_str());
    return false;
  }

  os << buffer;
  if (!os) {
    SHERPA_ONNX_LOGE("Write '%s' failed", filename.c_str());
    return false;
  }

  return true;
}

void WriteWave(char *buffer, int32_t sampling_rate, const float *samples_ch0,
               const float *samples_ch1, int32_t n) {
  WaveHeader header{};
  header.chunk_id = 0x46464952;      // FFIR
  header.format = 0x45564157;        // EVAW
  header.subchunk1_id = 0x20746d66;  // "fmt "
  header.subchunk1_size = 16;        // 16 for PCM
  header.audio_format = 1;           // PCM =1

  int32_t num_channels = samples_ch1 == nullptr ? 1 : 2;
  int32_t bits_per_sample = 16;  // int16_t

  header.num_channels = num_channels;
  header.sample_rate = sampling_rate;
  header.byte_rate = sampling_rate * num_channels * bits_per_sample / 8;
  header.block_align = num_channels * bits_per_sample / 8;
  header.bits_per_sample = bits_per_sample;
  header.subchunk2_id = 0x61746164;  // atad
  header.subchunk2_size = n * num_channels * bits_per_sample / 8;

  header.chunk_size = 36 + header.subchunk2_size;

  std::vector<int16_t> samples_int16_ch0(n);
  for (int32_t i = 0; i != n; ++i) {
    samples_int16_ch0[i] = std::min<int32_t>(samples_ch0[i] * 32767, 32767);
  }

  std::vector<int16_t> samples_int16_ch1;
  if (samples_ch1) {
    samples_int16_ch1.resize(n);
    for (int32_t i = 0; i != n; ++i) {
      samples_int16_ch1[i] = std::min<int32_t>(samples_ch1[i] * 32767, 32767);
    }
  }

  memcpy(buffer, &header, sizeof(WaveHeader));

  if (samples_ch1 == nullptr) {
    memcpy(buffer + sizeof(WaveHeader), samples_int16_ch0.data(),
           n * sizeof(int16_t));
  } else {
    auto p = reinterpret_cast<int16_t *>(buffer + sizeof(WaveHeader));

    for (int32_t i = 0; i != n; ++i) {
      p[2 * i] = samples_int16_ch0[i];
      p[2 * i + 1] = samples_int16_ch1[i];
    }
  }
}

}  // namespace sherpa_onnx
