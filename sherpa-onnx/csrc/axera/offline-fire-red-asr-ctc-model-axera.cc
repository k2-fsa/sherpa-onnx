// sherpa-onnx/csrc/axera/offline-fire-red-asr-ctc-model-axera.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/offline-fire-red-asr-ctc-model-axera.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "Eigen/Dense"
#include "ax_engine_api.h"  // NOLINT
#include "sherpa-onnx/csrc/axera/ax-engine-guard.h"
#include "sherpa-onnx/csrc/axera/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

class OfflineFireRedAsrCtcModelAxera::Impl {
 public:
  ~Impl() {
    FreeIO(&io_data_);
    if (handle_) {
      AX_ENGINE_DestroyHandle(handle_);
    }
  }

  explicit Impl(const OfflineModelConfig &config)
      : config_(config), allocator_{} {
    auto buf = ReadFile(config_.fire_red_asr_ctc.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config), allocator_{} {
    auto buf = ReadFile(mgr, config_.fire_red_asr_ctc.model);
    Init(buf.data(), buf.size());
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto features_shape = features.GetTensorTypeAndShapeInfo().GetShape();
    int32_t batch_size = features_shape[0];
    int32_t num_frames = features_shape[1];
    int32_t feat_dim = features_shape[2];

    const float *p_features = features.GetTensorData<float>();
    const int64_t *p_features_length = features_length.GetTensorData<int64_t>();

    if (batch_size != 1) {
      SHERPA_ONNX_LOGE("Only batch size 1 is supported by axera. Given: %d",
                       batch_size);
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t expected_frames = io_info_->pInputs[0].pShape[1];

    int32_t valid_frames = std::min<int32_t>(num_frames, expected_frames);
    valid_frames = std::min<int32_t>(valid_frames,
                                     static_cast<int32_t>(p_features_length[0]));

    std::vector<float> padded_features(expected_frames * feat_dim, 0.0f);
    std::copy(p_features, p_features + valid_frames * feat_dim,
              padded_features.begin());

    std::vector<int32_t> speech_length = {valid_frames};

    const auto &in0_meta = io_info_->pInputs[0];
    size_t bytes0 = in0_meta.nSize;
    if (bytes0 != padded_features.size() * sizeof(float)) {
      SHERPA_ONNX_LOGE(
          "Feature size mismatch. model expects %u bytes, but got %zu bytes",
          in0_meta.nSize, padded_features.size() * sizeof(float));
      SHERPA_ONNX_EXIT(-1);
    }

    std::memcpy(io_data_.pInputs[0].pVirAddr, padded_features.data(), bytes0);

    const auto &in1_meta = io_info_->pInputs[1];
    size_t bytes1 = in1_meta.nSize;
    if (bytes1 != speech_length.size() * sizeof(int32_t)) {
      SHERPA_ONNX_LOGE(
          "Speech length size mismatch. model expects %u bytes, but got %zu "
          "bytes",
          in1_meta.nSize, speech_length.size() * sizeof(int32_t));
      SHERPA_ONNX_EXIT(-1);
    }

    std::memcpy(io_data_.pInputs[1].pVirAddr, speech_length.data(), bytes1);

    auto ret = AX_ENGINE_RunSync(handle_, &io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out0_meta = io_info_->pOutputs[0];
    const auto &out0_buf = io_data_.pOutputs[0];

    int32_t out_frames = out0_meta.pShape[1];
    int32_t vocab_size = out0_meta.pShape[2];

    std::array<int64_t, 3> logits_shape = {1, out_frames, vocab_size};
    Ort::Value logits = Ort::Value::CreateTensor<float>(
        allocator_, logits_shape.data(), logits_shape.size());

    float *p_logits = logits.GetTensorMutableData<float>();
    std::memcpy(p_logits, out0_buf.pVirAddr, out0_meta.nSize);

    const auto &out1_meta = io_info_->pOutputs[1];
    const auto &out1_buf = io_data_.pOutputs[1];

    int64_t out_length = 0;
    if (out1_meta.eDataType == AX_ENGINE_DT_SINT32) {
      out_length = static_cast<int64_t>(
          reinterpret_cast<const int32_t *>(out1_buf.pVirAddr)[0]);
    } else if (out1_meta.eDataType == AX_ENGINE_DT_UINT32) {
      out_length = static_cast<int64_t>(
          reinterpret_cast<const uint32_t *>(out1_buf.pVirAddr)[0]);
    } else if (out1_meta.eDataType == AX_ENGINE_DT_FLOAT32) {
      out_length = static_cast<int64_t>(
          reinterpret_cast<const float *>(out1_buf.pVirAddr)[0]);
    } else {
      SHERPA_ONNX_LOGE("Unsupported length output dtype: %d",
                       static_cast<int32_t>(out1_meta.eDataType));
      SHERPA_ONNX_EXIT(-1);
    }

    std::array<int64_t, 1> lengths_shape = {1};
    Ort::Value lengths = Ort::Value::CreateTensor<int64_t>(
        allocator_, lengths_shape.data(), lengths_shape.size());

    int64_t *p_lengths = lengths.GetTensorMutableData<int64_t>();
    p_lengths[0] = out_length;

    std::vector<Ort::Value> ans;
    ans.push_back(std::move(logits));
    ans.push_back(std::move(lengths));

    return ans;
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }

  OrtAllocator *Allocator() { return allocator_; }

  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const {
    if (static_cast<int32_t>(mean_.size()) != feat_dim) {
      SHERPA_ONNX_LOGE("Bad things happened");
      SHERPA_ONNX_LOGE("Wrong feat dim %d. Expect: %d", feat_dim,
                       static_cast<int32_t>(mean_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    using RowMajorMat =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<RowMajorMat> x(features, num_frames, feat_dim);

    Eigen::Map<const Eigen::RowVectorXf> mean(mean_.data(), feat_dim);
    Eigen::Map<const Eigen::RowVectorXf> inv_std(inv_stddev_.data(), feat_dim);
    x.array() =
        (x.array().rowwise() - mean.array()).rowwise() * inv_std.array();
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &handle_);

    InitInputOutputAttrs(handle_, config_.debug, &io_info_);

    PrepareIO(io_info_, &io_data_, config_.debug);

    if (!io_info_ || io_info_->nInputSize != 2 || !io_info_->pInputs) {
      SHERPA_ONNX_LOGE("Axera FireRedASR CTC model expects 2 input tensors.");
      SHERPA_ONNX_EXIT(-1);
    }

    if (!io_info_->pOutputs || io_info_->nOutputSize != 2) {
      SHERPA_ONNX_LOGE(
          "Axera FireRedASR CTC model expects 2 output tensors.");
      SHERPA_ONNX_EXIT(-1);
    }

    if (io_info_->pOutputs[0].nShapeSize < 3) {
      SHERPA_ONNX_LOGE(
          "The first output tensor rank is too small (nShapeSize = %u)",
          io_info_->pOutputs[0].nShapeSize);
      SHERPA_ONNX_EXIT(-1);
    }

    subsampling_factor_ = 4;
    vocab_size_ = io_info_->pOutputs[0].pShape[io_info_->pOutputs[0].nShapeSize -
                                                1];

    if (config_.debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("subsampling_factor: %{public}d", subsampling_factor_);
      SHERPA_ONNX_LOGE("vocab_size: %{public}d", vocab_size_);
#else
      SHERPA_ONNX_LOGE("subsampling_factor: %d", subsampling_factor_);
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
#endif
    }

    mean_ = {10.498912811279297, 10.948603630065918, 11.889163970947266,
             12.634881973266602, 13.397452354431152, 14.010934829711914,
             14.450813293457031, 14.649748802185059, 14.791581153869629,
             14.72234058380127,  14.802156448364258, 14.86101245880127,
             15.077230453491211, 15.26024341583252,  15.328754425048828,
             15.397353172302246, 15.395853996276855, 15.34103775024414,
             15.4662446975708,   15.271865844726562, 15.108253479003906,
             15.295886993408203, 15.07359504699707,  15.177886009216309,
             15.0756254196167,   15.154109001159668, 15.051127433776855,
             15.130733489990234, 15.090286254882812, 15.099433898925781,
             15.128166198730469, 15.123964309692383, 15.144022941589355,
             15.198014259338379, 15.251392364501953, 15.329950332641602,
             15.4017972946167,   15.45089340209961,  15.500616073608398,
             15.435726165771484, 15.51086139678955,  15.44755744934082,
             15.510979652404785, 15.491739273071289, 15.538031578063965,
             15.608367919921875, 15.694382667541504, 15.762181282043457,
             15.821470260620117, 15.901959419250488, 15.907241821289062,
             15.925711631774902, 15.952259063720703, 16.000732421875,
             16.030330657958984, 16.060592651367188, 16.09003448486328,
             16.100107192993164, 16.091808319091797, 16.062585830688477,
             16.05771255493164,  15.997002601623535, 15.946383476257324,
             15.865278244018555, 15.778145790100098, 15.67629623413086,
             15.569791793823242, 15.515979766845703, 15.472077369689941,
             15.423379898071289, 15.382068634033203, 15.345854759216309,
             15.301891326904297, 15.26984691619873,  15.165450096130371,
             15.004508972167969, 14.87544059753418,  14.564188003540039,
             14.031693458557129, 13.159259796142578};
    inv_stddev_ = {
        0.2522108852863312,  0.23741021752357483, 0.23185651004314423,
        0.23331022262573242, 0.23203925788402557, 0.22906658053398132,
        0.22519451379776,    0.22010253369808197, 0.21958276629447937,
        0.22198699414730072, 0.22393390536308289, 0.22370608150959015,
        0.22321352362632751, 0.2220749408006668,  0.22118520736694336,
        0.22136786580085754, 0.2220366895198822,  0.222808837890625,
        0.22362081706523895, 0.224283829331398,   0.22464141249656677,
        0.22580783069133759, 0.22700978815555573, 0.22852766513824463,
        0.22993983328342438, 0.23110738396644592, 0.23227347433567047,
        0.23270530998706818, 0.23330524563789368, 0.23406001925468445,
        0.23448589444160461, 0.23556077480316162, 0.23632891476154327,
        0.23703691363334656, 0.2377307415008545,  0.23786373436450958,
        0.2380155622959137,  0.23858875036239624, 0.23943373560905457,
        0.2399062216281891,  0.24094033241271973, 0.24173252284526825,
        0.24236661195755005, 0.2430112659931183,  0.24341483414173126,
        0.243240088224411,   0.24262498319149017, 0.24218837916851044,
        0.24165891110897064, 0.241318941116333,   0.2413933277130127,
        0.24139994382858276, 0.241432324051857,   0.24122384190559387,
        0.24079066514968872, 0.24032147228717804, 0.24016834795475006,
        0.24034327268600464, 0.24069449305534363, 0.24123424291610718,
        0.24136029183864594, 0.24150611460208893, 0.24179506301879883,
        0.24160170555114746, 0.24221885204315186, 0.24253536760807037,
        0.24262426793575287, 0.2428186535835266,  0.24223484098911285,
        0.24199971556663513, 0.24160003662109375, 0.24074721336364746,
        0.23965489864349365, 0.23850350081920624, 0.2359732687473297,
        0.23006057739257812, 0.22904986143112183, 0.22814501821994781,
        0.22893856465816498, 0.23093441128730774};
  }

 private:
  std::mutex mutex_;
  AxEngineGuard ax_engine_guard_;

  OfflineModelConfig config_;
  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *io_info_ = nullptr;
  AX_ENGINE_IO_T io_data_;
  Ort::AllocatorWithDefaultOptions allocator_;

  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 0;

  std::vector<float> mean_;
  std::vector<float> inv_stddev_;
};

OfflineFireRedAsrCtcModelAxera::OfflineFireRedAsrCtcModelAxera(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineFireRedAsrCtcModelAxera::OfflineFireRedAsrCtcModelAxera(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineFireRedAsrCtcModelAxera::~OfflineFireRedAsrCtcModelAxera() = default;

std::vector<Ort::Value> OfflineFireRedAsrCtcModelAxera::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineFireRedAsrCtcModelAxera::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OfflineFireRedAsrCtcModelAxera::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

OrtAllocator *OfflineFireRedAsrCtcModelAxera::Allocator() const {
  return impl_->Allocator();
}

void OfflineFireRedAsrCtcModelAxera::NormalizeFeatures(float *features,
                                                       int32_t num_frames,
                                                       int32_t feat_dim) const {
  return impl_->NormalizeFeatures(features, num_frames, feat_dim);
}

#if __ANDROID_API__ >= 9
template OfflineFireRedAsrCtcModelAxera::OfflineFireRedAsrCtcModelAxera(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineFireRedAsrCtcModelAxera::OfflineFireRedAsrCtcModelAxera(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx

