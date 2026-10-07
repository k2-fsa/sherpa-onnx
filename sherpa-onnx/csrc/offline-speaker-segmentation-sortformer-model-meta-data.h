// sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model-meta-data.h
//
// Copyright (c)  2026  Silvio Tomatis

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_SORTFORMER_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_SORTFORMER_MODEL_META_DATA_H_

#include <cstdint>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/sortformer-speaker-cache.h"

namespace sherpa_onnx {

// If you are not sure what each field means, please
// have a look of the Python file in the model directory that
// you have downloaded.
//
// See scripts/nemo/nemotron-3-diarization/export_onnx.py
struct OfflineSpeakerSegmentationSortformerModelMetaData {
  // Frontend: log-mel features computed like NeMo, without normalization
  int32_t sample_rate = 16000;
  int32_t n_fft = 512;
  int32_t win_length = 400;
  int32_t hop_length = 160;
  int32_t num_mel_bins = 128;
  float preemphasis = 0.97f;

  // The following are in encoder frames, i.e., every subsampling_factor
  // feature frames
  int32_t chunk_length = 340;
  int32_t chunk_right_context = 40;

  // num_speakers, subsampling_factor, hidden_size, fifo_length, etc.
  SortformerSpeakerCacheConfig cache;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_SEGMENTATION_SORTFORMER_MODEL_META_DATA_H_
