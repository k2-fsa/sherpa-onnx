// sherpa-onnx/csrc/ascend/offline-paraformer-model-ascend.cc
//
// Copyright (c)  2025  Xiaomi Corporation

// References:
// https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/appdevgapi/aclcppdevg_03_0298.html
#include <algorithm>
#include <array>
#include <mutex>  // NOLINT
#include <vector>

#include "sherpa-onnx/csrc/ascend/offline-paraformer-model-ascend.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/ascend/macros.h"
#include "sherpa-onnx/csrc/ascend/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineParaformerModelAscend::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    PreInit();

    std::vector<std::string> filenames;
    SplitStringToVector(config_.paraformer.model, ",", false, &filenames);
    if (filenames.size() != 3) {
      SHERPA_ONNX_LOGE("Invalid paraformer ascend NPU model '%s'",
                       config_.paraformer.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    InitEncoder(filenames[0]);
    InitPredictor(filenames[1]);
    InitDecoder(filenames[2]);

    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    PreInit();

    std::vector<std::string> filenames;
    SplitStringToVector(config_.paraformer.model, ",", false, &filenames);
    if (filenames.size() != 3) {
      SHERPA_ONNX_LOGE("Invalid paraformer ascend NPU model '%s'",
                       config_.paraformer.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    {
      auto buf = ReadFile(mgr, filenames[0]);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, filenames[1]);
      InitPredictor(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, filenames[2]);
      InitDecoder(buf.data(), buf.size());
    }

    PostInit();
  }

  const OfflineParaformerModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) {
    // TODO(fangjun): Support multi clients
    std::lock_guard<std::mutex> lock(mutex_);

    features = ApplyLFR(std::move(features));

    int32_t num_frames = features.size() / 560;

    aclError ret =
        aclrtMemcpy(*x_ptr_, features.size() * sizeof(float), features.data(),
                    features.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    std::array<int32_t, 4> prompt_array{language, 1, 2, text_norm};
    ret = aclrtMemcpy(*prompt_ptr_, prompt_ptr_->Size(), prompt_array.data(),
                      prompt_ptr_->Size(), ACL_MEMCPY_HOST_TO_DEVICE);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    AclMdlDataset input_dataset;
    AclDataBuffer x_buf(*x_ptr_, features.size() * sizeof(float));
    input_dataset.AddBuffer(x_buf);

    AclDataBuffer prompt_buf(*prompt_ptr_, prompt_ptr_->Size());
    input_dataset.AddBuffer(prompt_buf);

    // 动态Shape输入（设置Shape范围）
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/appdevg/acldevg/aclcppdevg_000044.html

    std::array<int64_t, 3> x_shape = {1, num_frames, 560};
    AclTensorDesc x_desc(ACL_FLOAT, x_shape.size(), x_shape.data(),
                         ACL_FORMAT_ND);
    input_dataset.SetTensorDesc(x_desc, 0);

    std::array<int64_t, 1> prompt_shape = {4};
    AclTensorDesc prompt_desc(ACL_INT32, prompt_shape.size(),
                              prompt_shape.data(), ACL_FORMAT_ND);
    input_dataset.SetTensorDesc(prompt_desc, 1);

    AclMdlDataset output_dataset;

    AclDataBuffer logits_buf(*logits_ptr_,
                             num_frames * vocab_size_ * sizeof(float));
    output_dataset.AddBuffer(logits_buf);

    ret = aclmdlExecute(*model_, input_dataset, output_dataset);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlExecute");

    std::vector<float> logits(num_frames * vocab_size_);
    ret = aclrtMemcpy(logits.data(), num_frames * vocab_size_ * sizeof(float),
                      *logits_ptr_, num_frames * vocab_size_ * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMemcpy");

    return logits;
  }

 private:
  void InitEncoder(const std::string &filename) {
    encoder_model_ = std::make_unique<AclModel>(filename);
    if (config_.debug) {
      auto s = encoder_model_->GetInfo();

      SHERPA_ONNX_LOGE("----encoder----\n%s\n", s.c_str());
    }
  }

  void InitPredictor(const std::string &filename) {
    predictor_model_ = std::make_unique<AclModel>(filename);
    if (config_.debug) {
      auto s = predictor_model_->GetInfo();

      SHERPA_ONNX_LOGE("----predictor----\n%s\n", s.c_str());
    }
  }

  void InitDecoder(const std::string &filename) {
    decoder_model_ = std::make_unique<AclModel>(filename);
    if (config_.debug) {
      auto s = decoder_model_->GetInfo();

      SHERPA_ONNX_LOGE("----decoder----\n%s\n", s.c_str());
    }
  }

  void InitEncoder(void *data, size_t size) {
    encoder_model_ = std::make_unique<AclModel>(data, size);
    if (config_.debug) {
      auto s = model_->GetInfo();
      SHERPA_ONNX_LOGE("----encoder----\n%s\n", s.c_str());
    }
  }

  void InitPredictor(void *data, size_t size) {
    predictor_model_ = std::make_unique<AclModel>(data, size);
    if (config_.debug) {
      auto s = model_->GetInfo();
      SHERPA_ONNX_LOGE("----predictor----\n%s\n", s.c_str());
    }
  }

  void InitDecoder(void *data, size_t size) {
    decoder_model_ = std::make_unique<AclModel>(data, size);
    if (config_.debug) {
      auto s = model_->GetInfo();
      SHERPA_ONNX_LOGE("----decoder----\n%s\n", s.c_str());
    }
  }

  void PreInit() {
    int32_t device_id = 0;
    aclError ret = aclrtSetDevice(device_id);
    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to call aclrtSetDevice with device id: %d", device_id);

    context_ = std::make_unique<AclContext>(device_id);

    ret = aclrtSetCurrentContext(*context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtSetCurrentContext");
  }

  void PostInit() {
    encoder_dim_ = encoder_model_->GetOutputShapes()[0].back();
    vocab_size_ = deocder_model_->GetOutputShapes()[0].back();

    Preallocate();
  }

  void Preallocate() {
    // max 30 seconds
    max_num_frames_ = (30 * 100 - 7) / 6 + 1;

    features_ptr_ = std::make_unique<AclDevicePtr>(max_num_frames_ * feat_dim_ *
                                                   sizeof(float));

    encoder_out_ptr_ = std::make_unique<AclDevicePtr>(
        max_num_frames_ * encoder_dim_ * sizeof(float));

    alphas_ptr_ = std::make_unique<AclDevicePtr>(max_num_frames_ *
                                                 encoder_dim_ * sizeof(float));

    acoustic_embedding_ptr_ = std::make_unique<AclDevicePtr>(
        max_num_frames_ * encoder_dim_ * sizeof(float));

    logits_ptr_ = std::make_unique<AclDevicePtr>(max_num_frames_ * vocab_size_ *
                                                 sizeof(float));
  }

  std::vector<float> ApplyLFR(std::vector<float> in) const {
    int32_t lfr_window_size = meta_data_.window_size;
    int32_t lfr_window_shift = meta_data_.window_shift;
    int32_t in_feat_dim = 80;

    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;

    if (out_num_frames > max_num_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          out_num_frames, max_num_frames_);

      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios.");

      out_num_frames = max_num_frames_;
    }

    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(out_num_frames * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

 private:
  std::mutex mutex_;
  Acl acl_;

  std::unique_ptr<AclContext> context_;

  OfflineModelConfig config_;

  std::unique_ptr<AclModel> encoder_model_;
  std::unique_ptr<AclModel> predictor_model_;
  std::unique_ptr<AclModel> decoder_model_;

  int32_t encoder_dim_ = 0;
  int32_t vocab_size_ = 0;
  int32_t max_num_frames_ = 0;
  int32_t feat_dim_ = 560;

  std::unique_ptr<AclDevicePtr> features_ptr_;
  std::unique_ptr<AclDevicePtr> encoder_out_ptr_;
  std::unique_ptr<AclDevicePtr> alphas_ptr_;
  std::unique_ptr<AclDevicePtr> acoustic_embedding_ptr_;
  std::unique_ptr<AclDevicePtr> logits_ptr_;
};

OfflineParaformerModelAscend::OfflineParaformerModelAscend(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineParaformerModelAscend::~OfflineParaformerModelAscend() = default;

std::vector<float> OfflineParaformerModelAscend::Run(
    std::vector<float> features) const {
  return impl_->Run(std::move(features));
}

#if __ANDROID_API__ >= 9
template OfflineParaformerModelAscend::OfflineParaformerModelAscend(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineParaformerModelAscend::OfflineParaformerModelAscend(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
