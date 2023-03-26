// sherpa-onnx/csrc/wave-reader.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/wave-reader.h"

#include <cassert>
#include <fstream>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {
namespace {
// see http://soundfile.sapp.org/doc/WaveFormat/
//
// Note: We assume little endian here
// TODO(fangjun): Support big endian
struct WaveHeader {
  bool Validate() const {
    //                 F F I R
    if (chunk_id != 0x46464952) {
      SHERPA_ONNX_LOGE("Expected chunk_id RIFF. Given: 0x%08x\n", chunk_id);
      return false;
    }
    //               E V A W
    if (format != 0x45564157) {
      SHERPA_ONNX_LOGE("Expected format WAVE. Given: 0x%08x\n", format);
      return false;
    }

    if (subchunk1_id != 0x20746d66) {
      SHERPA_ONNX_LOGE("Expected subchunk1_id 0x20746d66. Given: 0x%08x\n",
                       subchunk1_id);
      return false;
    }

    if (subchunk1_size != 16) {  // 16 for PCM
      SHERPA_ONNX_LOGE("Expected subchunk1_size 16. Given: %d\n",
                       subchunk1_size);
      return false;
    }

    if (audio_format != 1) {  // 1 for PCM
      SHERPA_ONNX_LOGE("Expected audio_format 1. Given: %d\n", audio_format);
      return false;
    }

    if (num_channels != 1) {  // we support only single channel for now
      SHERPA_ONNX_LOGE("Expected single channel. Given: %d\n", num_channels);
      return false;
    }
    if (byte_rate != (sample_rate * num_channels * bits_per_sample / 8)) {
      return false;
    }

    if (block_align != (num_channels * bits_per_sample / 8)) {
      return false;
    }

    if (bits_per_sample != 16) {  // we support only 16 bits per sample
      SHERPA_ONNX_LOGE("Expected bits_per_sample 16. Given: %d\n",
                       bits_per_sample);
      return false;
    }

    return true;
  }

  // See
  // https://en.wikipedia.org/wiki/WAV#Metadata
  // and
  // https://www.robotplanet.dk/audio/wav_meta_data/riff_mci.pdf
  void SeekToDataChunk(std::istream &is) {
    //                              a t a d
    while (is && subchunk2_id != 0x61746164) {
      // const char *p = reinterpret_cast<const char *>(&subchunk2_id);
      // printf("Skip chunk (%x): %c%c%c%c of size: %d\n", subchunk2_id, p[0],
      //        p[1], p[2], p[3], subchunk2_size);
      is.seekg(subchunk2_size, std::istream::cur);
      is.read(reinterpret_cast<char *>(&subchunk2_id), sizeof(int32_t));
      is.read(reinterpret_cast<char *>(&subchunk2_size), sizeof(int32_t));
    }
  }

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
static_assert(sizeof(WaveHeader) == 44, "");

// Read a wave file of mono-channel.
// Return its samples normalized to the range [-1, 1).
std::vector<float> ReadWaveImpl(std::istream &is, int32_t *sampling_rate,
                                bool *is_ok) {
  WaveHeader header;
  is.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!is) {
    *is_ok = false;
    return {};
  }

  if (!header.Validate()) {
    *is_ok = false;
    return {};
  }

  header.SeekToDataChunk(is);
  if (!is) {
    *is_ok = false;
    return {};
  }

  *sampling_rate = header.sample_rate;

  // header.subchunk2_size contains the number of bytes in the data.
  // As we assume each sample contains two bytes, so it is divided by 2 here
  std::vector<int16_t> samples(header.subchunk2_size / 2);

  is.read(reinterpret_cast<char *>(samples.data()), header.subchunk2_size);
  if (!is) {
    *is_ok = false;
    return {};
  }

  std::vector<float> ans(samples.size());
  for (int32_t i = 0; i != ans.size(); ++i) {
    ans[i] = samples[i] / 32768.;
  }

  *is_ok = true;
  return ans;
}

}  // namespace

std::vector<float> ReadWave(const std::string &filename, int32_t *sampling_rate,
                            bool *is_ok) {
  std::ifstream is(filename, std::ifstream::binary);
  return ReadWave(is, sampling_rate, is_ok);
}

std::vector<float> ReadWave(std::istream &is, int32_t *sampling_rate,
                            bool *is_ok) {
  auto samples = ReadWaveImpl(is, sampling_rate, is_ok);
  return samples;
}

}  // namespace sherpa_onnx
