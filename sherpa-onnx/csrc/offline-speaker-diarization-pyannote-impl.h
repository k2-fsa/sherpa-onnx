// sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/fast-clustering.h"
#include "sherpa-onnx/csrc/offline-speaker-diarization-impl.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

namespace sherpa_onnx {

using Matrix2D =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using Matrix2DInt32 =
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using FloatRowVector = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using Int32RowVector = Eigen::Matrix<int32_t, 1, Eigen::Dynamic>;

using Int32Pair = std::pair<int32_t, int32_t>;

class OfflineSpeakerDiarizationPyannoteImpl
    : public OfflineSpeakerDiarizationImpl {
 public:
  ~OfflineSpeakerDiarizationPyannoteImpl() override = default;

  explicit OfflineSpeakerDiarizationPyannoteImpl(
      const OfflineSpeakerDiarizationConfig &config)
      : config_(config),
        segmentation_model_(config_.segmentation),
        embedding_extractor_(config_.embedding),
        clustering_(config_.clustering) {
    Init();
  }

  OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const override {
    std::vector<Matrix2D> segmentations = RunSpeakerSegmentationModel(audio, n);
    // segmentations[i] is for chunk_i
    // Each matrix is of shape (num_frames, num_powerset_classes)
    if (segmentations.empty()) {
      return {};
    }

    std::cout << "segmentations.size() " << segmentations.size() << "---"
              << segmentations[0].rows() << ", " << segmentations[1].cols()
              << "\n";

    std::vector<Matrix2DInt32> labels;
    labels.reserve(segmentations.size());

    for (const auto &m : segmentations) {
      labels.push_back(ToMultiLabel(m));
    }

    segmentations.clear();

    // labels[i] is a 0-1 matrix of shape (num_frames, num_speakers)

    // speaker count per frame
    Int32RowVector speaker_count = ComputeSpeakerCount(labels);
    std::cout << "speaker count: " << speaker_count.cast<float>().sum() << ", "
              << speaker_count.cast<float>().mean() << "\n";

    if (speaker_count.maxCoeff() == 0) {
      SHERPA_ONNX_LOGE("No speakers found in the audio samples");
      return {};
    }

    auto chunk_speaker_samples_list_pair = GetChunkSpeakerSampleIndexes(labels);
    Matrix2D embeddings =
        ComputeEmbeddings(audio, n, chunk_speaker_samples_list_pair.second,
                          callback, callback_arg);

    std::vector<int32_t> cluster_labels = clustering_.Cluster(
        &embeddings(0, 0), embeddings.rows(), embeddings.cols());

    for (int32_t i = 0; i != cluster_labels.size(); ++i) {
      std::cout << i << "->" << cluster_labels[i] << "\n";
    }

    return {};
  }

 private:
  void Init() { InitPowersetMapping(); }

  // see also
  // https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/utils/powerset.py#L68
  void InitPowersetMapping() {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t num_classes = meta_data.num_classes;
    int32_t powerset_max_classes = meta_data.powerset_max_classes;
    int32_t num_speakers = meta_data.num_speakers;

    powerset_mapping_ = Matrix2DInt32(num_classes, num_speakers);
    powerset_mapping_.setZero();

    int32_t k = 1;
    for (int32_t i = 1; i <= powerset_max_classes; ++i) {
      if (i == 1) {
        for (int32_t j = 0; j != num_speakers; ++j, ++k) {
          powerset_mapping_(k, j) = 1;
        }
      } else if (i == 2) {
        for (int32_t j = 0; j != num_speakers; ++j) {
          for (int32_t m = j + 1; m < num_speakers; ++m, ++k) {
            powerset_mapping_(k, j) = 1;
            powerset_mapping_(k, m) = 1;
          }
        }
      } else {
        SHERPA_ONNX_LOGE(
            "powerset_max_classes = %d is currently not supported!", i);
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  std::vector<Matrix2D> RunSpeakerSegmentationModel(const float *audio,
                                                    int32_t n) const {
    std::vector<Matrix2D> ans;

    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;

    if (n <= 0) {
      SHERPA_ONNX_LOGE(
          "number of audio samples is %d (<= 0). Please provide a positive "
          "number",
          n);
      return {};
    }

    if (n <= window_size) {
      std::vector<float> buf(window_size);
      // NOTE: buf is zero initialized by default

      std::copy(audio, audio + n, buf.data());

      Matrix2D m = ProcessChunk(buf.data());

      ans.push_back(std::move(m));

      return ans;
    }

    int32_t num_chunks = (n - window_size) / window_shift + 1;
    bool has_last_chunk = (n - window_size) % window_shift > 0;

    ans.reserve(num_chunks + has_last_chunk);

    const float *p = audio;

    for (int32_t i = 0; i != num_chunks; ++i, p += window_shift) {
      Matrix2D m = ProcessChunk(p);

      ans.push_back(std::move(m));
    }

    if (has_last_chunk) {
      std::vector<float> buf(window_size);
      std::copy(p, audio + n, buf.data());

      Matrix2D m = ProcessChunk(buf.data());

      ans.push_back(std::move(m));
    }

    return ans;
  }

  Matrix2D ProcessChunk(const float *p) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> shape = {1, 1, window_size};

    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, const_cast<float *>(p),
                                 window_size, shape.data(), shape.size());

    Ort::Value out = segmentation_model_.Forward(std::move(x));
    std::vector<int64_t> out_shape = out.GetTensorTypeAndShapeInfo().GetShape();
    Matrix2D m(out_shape[1], out_shape[2]);
    std::copy(out.GetTensorData<float>(), out.GetTensorData<float>() + m.size(),
              &m(0, 0));
    return m;
  }

  Matrix2DInt32 ToMultiLabel(const Matrix2D &m) const {
    int32_t num_rows = m.rows();
    Matrix2DInt32 ans(num_rows, powerset_mapping_.cols());

    std::ptrdiff_t col_id;

    for (int32_t i = 0; i != num_rows; ++i) {
      m.row(i).maxCoeff(&col_id);
      ans.row(i) = powerset_mapping_.row(col_id);
    }

    std::cout << "sum labels: " << ans.colwise().sum() << "\n";
    return ans;
  }

  // See also
  // https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/utils/diarization.py#L122
  Int32RowVector ComputeSpeakerCount(
      const std::vector<Matrix2DInt32> &labels) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    int32_t num_chunks = labels.size();

    int32_t num_frames = (window_size + (num_chunks - 1) * window_shift) /
                             receptive_field_shift +
                         1;

    FloatRowVector count(num_frames);
    FloatRowVector weight(num_frames);
    count.setZero();
    weight.setZero();

    for (int32_t i = 0; i != num_chunks; ++i) {
      int32_t start =
          static_cast<float>(i) * window_shift / receptive_field_shift + 0.5;

      auto seq = Eigen::seqN(start, labels[i].rows());

      count(seq).array() += labels[i].rowwise().sum().array().cast<float>();

      weight(seq).array() += 1;
    }

    return ((count.array() / (weight.array() + 1e-12f)) + 0.5).cast<int32_t>();
  }

  // ans.first: a list of (chunk_id, speaker_id)
  // ans.second: a list of list of (start_sample_index, end_sample_index)
  //
  // ans.first[i] corresponds to ans.second[i]
  std::pair<std::vector<Int32Pair>, std::vector<std::vector<Int32Pair>>>
  GetChunkSpeakerSampleIndexes(const std::vector<Matrix2DInt32> &labels) const {
    auto new_labels = ExcludeOverlap(labels);

    std::vector<Int32Pair> chunk_speaker_list;
    std::vector<std::vector<Int32Pair>> samples_index_list;

    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;
    int32_t num_speakers = meta_data.num_speakers;

    int32_t chunk_index = 0;
    for (const auto &label : new_labels) {
      Matrix2DInt32 tmp = label.transpose();
      // tmp: (num_speakers, num_frames)
      int32_t num_frames = tmp.cols();

      int32_t sample_offset = chunk_index * window_shift;

      for (int32_t speaker_index = 0; speaker_index != num_speakers;
           ++speaker_index) {
        auto d = tmp.row(speaker_index);
        if (d.sum() < 10) {
          // skip segments less than 10 frames
          continue;
        }

        Int32Pair this_chunk_speaker = {chunk_index, speaker_index};
        std::vector<Int32Pair> this_speaker_samples;

        bool started = false;
        int32_t start_index;

        for (int32_t k = 0; k != num_frames; ++k) {
          if (d[k] != 0) {
            if (!started) {
              started = true;
              start_index = k;
            }
          } else if (started) {
            started = false;

            int32_t start_samples =
                static_cast<float>(start_index) / num_frames * window_size +
                sample_offset;
            int32_t end_samples =
                static_cast<float>(k) / num_frames * window_size +
                sample_offset;

            this_speaker_samples.emplace_back(start_samples, end_samples);
          }
        }

        if (started) {
          int32_t start_samples =
              static_cast<float>(start_index) / num_frames * window_size +
              sample_offset;
          int32_t end_samples =
              static_cast<float>(num_frames - 1) / num_frames * window_size +
              sample_offset;
          this_speaker_samples.emplace_back(start_samples, end_samples);
        }

        chunk_speaker_list.push_back(std::move(this_chunk_speaker));
        samples_index_list.push_back(std::move(this_speaker_samples));
      }  // for (int32_t speaker_index = 0;
      chunk_index += 1;
    }  // for (const auto &label : new_labels)

    return {chunk_speaker_list, samples_index_list};
  }

  // If there are multiple speakers at a frame, then this frame is excluded.
  std::vector<Matrix2DInt32> ExcludeOverlap(
      const std::vector<Matrix2DInt32> &labels) const {
    int32_t num_chunks = labels.size();
    std::vector<Matrix2DInt32> ans;
    ans.reserve(num_chunks);

    for (const auto &label : labels) {
      Matrix2DInt32 new_label(label.rows(), label.cols());
      new_label.setZero();
      Int32RowVector v = label.rowwise().sum();

      for (int32_t i = 0; i != v.cols(); ++i) {
        if (v[i] < 2) {
          new_label.row(i) = label.row(i);
        }
      }

      ans.push_back(std::move(new_label));
    }

    return ans;
  }

  /**
   * @param sample_indexes[i] contains the sample segment start and end indexes
   *                          for the i-th (chunk, speaker) pair
   * @return Return a matrix of shape (sample_indexes.size(), embedding_dim)
   *         where ans.row[i] contains the embedding for the
   *         i-th (chunk, speaker) pair
   */
  Matrix2D ComputeEmbeddings(
      const float *audio, int32_t n,
      const std::vector<std::vector<Int32Pair>> &sample_indexes,
      OfflineSpeakerDiarizationProgressCallback callback,
      void *callback_arg) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t sample_rate = meta_data.sample_rate;
    Matrix2D ans(sample_indexes.size(), embedding_extractor_.Dim());

    int32_t k = 0;
    for (const auto &v : sample_indexes) {
      auto stream = embedding_extractor_.CreateStream();
      for (const auto &p : v) {
        int32_t end = (p.second <= n) ? p.second : n;
        int32_t num_samples = end - p.first;

        if (num_samples > 0) {
          stream->AcceptWaveform(sample_rate, audio + p.first, num_samples);
        }
      }

      stream->InputFinished();
      if (!embedding_extractor_.IsReady(stream.get())) {
        SHERPA_ONNX_LOGE(
            "This segment is too short, which should not happen since we have "
            "already filtered short segments");
        SHERPA_ONNX_EXIT(-1);
      }

      std::vector<float> embedding = embedding_extractor_.Compute(stream.get());

      std::copy(embedding.begin(), embedding.end(), &ans(k, 0));

      k += 1;

      if (callback) {
        callback(k, ans.rows(), callback_arg);
      }
    }

    return ans;
  }

 private:
  OfflineSpeakerDiarizationConfig config_;
  OfflineSpeakerSegmentationPyannoteModel segmentation_model_;
  SpeakerEmbeddingExtractor embedding_extractor_;
  FastClustering clustering_;
  Matrix2DInt32 powerset_mapping_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
