// sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/offline-speaker-diarization-impl.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.h"

namespace sherpa_onnx {

using Matrix2D =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class OfflineSpeakerDiarizationPyannoteImpl
    : public OfflineSpeakerDiarizationImpl {
 public:
  ~OfflineSpeakerDiarizationPyannoteImpl() override = default;

  explicit OfflineSpeakerDiarizationPyannoteImpl(
      const OfflineSpeakerDiarizationConfig &config)
      : config_(config), segmentation_model_(config_.segmentation) {}

  OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const override {
    std::vector<Matrix2D> segmentations =
        RunSpeakerSegmentationModel(audio, n, callback, callback_arg);
    // segmentations[i] is for chunk_i
    // Each matrix is of shape (num_frames, num_powerset_classes)
    if (segmentations.empty()) {
      return {};
    }

    std::cout << "segmentations.size() " << segmentations.size() << "\n";
    for (const auto &m : segmentations) {
      std::cout << m.rows() << ", " << m.cols() << "\n";
    }

    return {};
  }

 private:
  std::vector<Matrix2D> RunSpeakerSegmentationModel(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback,
      void *callback_arg) const {
    std::vector<Matrix2D> ans;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

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
      std::copy(audio, audio + n, buf.data());

      std::array<int64_t, 3> shape = {1, 1, window_size};

      Ort::Value x = Ort::Value::CreateTensor(
          memory_info, buf.data(), buf.size(), shape.data(), shape.size());

      Ort::Value out = segmentation_model_.Forward(std::move(x));
      std::vector<int64_t> out_shape =
          out.GetTensorTypeAndShapeInfo().GetShape();
      Matrix2D m(out_shape[1], out_shape[2]);
      std::copy(out.GetTensorData<float>(),
                out.GetTensorData<float>() + m.size(), &m(0, 0));

      ans.push_back(std::move(m));

      if (callback) {
        callback(1, 1, callback_arg);
      }

      return ans;
    }

    int32_t num_chunks = (n - window_size) / window_shift + 1;
    bool has_last_chunk = (n - window_size) % window_shift > 0;

    ans.reserve(num_chunks + has_last_chunk);

    std::array<int64_t, 3> shape = {1, 1, window_size};

    const float *p = audio;
    for (int32_t i = 0; i != num_chunks; ++i, p += window_shift) {
      Ort::Value x =
          Ort::Value::CreateTensor(memory_info, const_cast<float *>(p),
                                   window_size, shape.data(), shape.size());

      Ort::Value out = segmentation_model_.Forward(std::move(x));
      std::vector<int64_t> out_shape =
          out.GetTensorTypeAndShapeInfo().GetShape();

      Matrix2D m(out_shape[1], out_shape[2]);
      std::copy(out.GetTensorData<float>(),
                out.GetTensorData<float>() + m.size(), &m(0, 0));

      ans.push_back(std::move(m));

      if (callback) {
        callback(i + 1, num_chunks + has_last_chunk, callback_arg);
      }
    }

    if (has_last_chunk) {
      std::vector<float> buf(window_size);
      std::copy(p, audio + n, buf.data());

      Ort::Value x = Ort::Value::CreateTensor(
          memory_info, buf.data(), buf.size(), shape.data(), shape.size());

      Ort::Value out = segmentation_model_.Forward(std::move(x));
      std::vector<int64_t> out_shape =
          out.GetTensorTypeAndShapeInfo().GetShape();
      Matrix2D m(out_shape[1], out_shape[2]);
      std::copy(out.GetTensorData<float>(),
                out.GetTensorData<float>() + m.size(), &m(0, 0));

      ans.push_back(std::move(m));
      if (callback) {
        callback(num_chunks + 1, num_chunks + 1, callback_arg);
      }
    }

    return ans;
  }

 private:
  OfflineSpeakerDiarizationConfig config_;
  OfflineSpeakerSegmentationPyannoteModel segmentation_model_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
