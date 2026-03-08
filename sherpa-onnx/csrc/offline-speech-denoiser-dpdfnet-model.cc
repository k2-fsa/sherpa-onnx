// sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model.h"

#include <algorithm>
#include <array>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

namespace {

static constexpr int32_t kDpdfNet16kFreqBins = 161;
static constexpr int32_t kDpdfNet16kSampleRate = 16000;
static constexpr int32_t kDpdfNet16kErbBins = 32;
static constexpr int32_t kDpdfNet16kSpecBins = 96;

static constexpr int32_t kDpdfNet48kFreqBins = 481;
static constexpr int32_t kDpdfNet48kSampleRate = 48000;
static constexpr int32_t kDpdfNet48kSpecBins = 96;

static constexpr float kEmpiricalMagNormMu0[] = {
    -32.5573f, -34.8212f, -38.8536f, -39.8786f, -40.7435f, -41.5264f,
    -42.2550f, -42.4084f, -42.4310f, -42.6460f, -43.0335f, -43.6704f,
    -44.5702f, -45.3495f, -46.0085f, -46.4555f, -46.6586f, -46.5344f,
    -46.6155f, -46.8084f, -46.9929f, -47.1601f, -47.2527f, -47.3922f,
    -47.6403f, -47.9241f, -48.1503f, -48.3009f, -48.2301f, -48.3322f,
    -48.7560f, -48.9983f, -48.9335f, -48.9662f, -49.1459f, -49.3420f,
    -49.7211f, -50.0116f, -49.9385f, -49.9582f, -50.2630f, -50.5991f,
    -50.6968f, -50.6802f, -50.6985f, -50.7848f, -51.0159f, -51.2049f,
    -51.1635f, -51.1143f, -51.2253f, -51.3587f, -51.4638f, -51.6249f,
    -51.6959f, -51.6421f, -51.7896f, -52.1039f, -52.2228f, -52.2449f,
    -52.3748f, -52.4833f, -52.6428f, -52.8336f, -52.9246f, -52.9568f,
    -53.0166f, -53.0936f, -53.2231f, -53.3220f, -53.2919f, -53.2598f,
    -53.3251f, -53.4057f, -53.4346f, -53.4701f, -53.5049f, -53.5446f,
    -53.6280f, -53.7321f, -53.8563f, -53.9212f, -53.9847f, -54.0431f,
    -54.2167f, -54.4016f, -54.4896f, -54.5848f, -54.6639f, -54.7624f,
    -54.9358f, -55.1197f, -55.2275f, -55.3156f, -55.4165f, -55.5399f,
    -55.6764f, -55.7395f, -55.7477f, -55.8479f, -55.9744f, -56.0430f,
    -56.0860f, -56.1640f, -56.2363f, -56.2628f, -56.3778f, -56.4760f,
    -56.5116f, -56.5653f, -56.6614f, -56.7072f, -56.7463f, -56.8341f,
    -56.8720f, -56.9147f, -56.9865f, -57.0185f, -57.0088f, -57.0422f,
    -57.1057f, -57.1574f, -57.1939f, -57.2076f, -57.2134f, -57.2274f,
    -57.2744f, -57.3180f, -57.2798f, -57.3123f, -57.3425f, -57.3012f,
    -57.2734f, -57.2837f, -57.2924f, -57.2481f, -57.2324f, -57.2853f,
    -57.2801f, -57.2275f, -57.2241f, -57.2267f, -57.2223f, -57.2299f,
    -57.2213f, -57.1935f, -57.2167f, -57.2694f, -57.2692f, -57.2359f,
    -57.2661f, -57.2832f, -57.2845f, -57.2923f, -57.3002f, -57.3082f,
    -57.2977f, -57.3481f, -57.3689f, -57.3640f, -57.3757f, -57.4232f,
    -57.4438f, -57.4617f, -57.5147f, -57.5294f, -57.5081f, -57.5383f,
    -57.5932f, -57.5671f, -57.5515f, -57.5581f, -57.5631f, -57.5515f,
    -57.5555f, -57.5938f, -57.6046f, -57.6050f, -57.6177f, -57.6236f,
    -57.6322f, -57.6557f, -57.7025f, -57.7337f, -57.7679f, -57.8210f,
    -57.8595f, -57.8640f, -57.8880f, -57.9267f, -57.9714f, -57.9995f,
    -58.0165f, -58.0381f, -58.0720f, -58.1155f, -58.1525f, -58.1538f,
    -58.1753f, -58.2077f, -58.2278f, -58.2362f, -58.2209f, -58.2060f,
    -58.2296f, -58.2728f, -58.3094f, -58.3506f, -58.3813f, -58.4140f,
    -58.4446f, -58.4497f, -58.4721f, -58.4972f, -58.5253f, -58.5301f,
    -58.5328f, -58.5627f, -58.5960f, -58.6233f, -58.6604f, -58.7172f,
    -58.7706f, -58.8209f, -58.8839f, -58.9435f, -59.0027f, -59.0626f,
    -59.1240f, -59.2068f, -59.2783f, -59.3370f, -59.4002f, -59.4317f,
    -59.4730f, -59.4850f, -59.5219f, -59.5662f, -59.5398f, -59.5387f,
    -59.5882f, -59.6119f, -59.6049f, -59.6075f, -59.6520f, -59.6765f,
    -59.6948f, -59.7326f, -59.7967f, -59.8499f, -59.9053f, -59.9388f,
    -60.0229f, -60.0928f, -60.1238f, -60.1692f, -60.2615f, -60.3149f,
    -60.3411f, -60.4097f, -60.4845f, -60.5007f, -60.5397f, -60.6091f,
    -60.6559f, -60.6968f, -60.7423f, -60.8142f, -60.8643f, -60.9090f,
    -60.9342f, -60.9754f, -61.0166f, -61.0444f, -61.0889f, -61.1499f,
    -61.1811f, -61.2334f, -61.2779f, -61.3123f, -61.3728f, -61.4356f,
    -61.4621f, -61.5222f, -61.5875f, -61.6326f, -61.6533f, -61.7240f,
    -61.7653f, -61.7825f, -61.8327f, -61.8723f, -61.9088f, -61.9167f,
    -61.9362f, -62.0017f, -62.0458f, -62.0655f, -62.0833f, -62.1180f,
    -62.1346f, -62.1610f, -62.1685f, -62.2173f, -62.2716f, -62.3036f,
    -62.3412f, -62.3841f, -62.4318f, -62.4740f, -62.5171f, -62.5457f,
    -62.5202f, -62.6277f, -62.7108f, -62.7737f, -62.8414f, -62.8911f,
    -62.9369f, -62.9722f, -62.9966f, -63.0276f, -63.0892f, -63.1491f,
    -63.1747f, -63.2264f, -63.2771f, -63.2967f, -63.3073f, -63.3486f,
    -63.3913f, -63.4314f, -63.4603f, -63.4844f, -63.5416f, -63.5944f,
    -63.6229f, -63.6585f, -63.7060f, -63.7358f, -63.7645f, -63.7970f,
    -63.8262f, -63.8632f, -63.9072f, -63.9406f, -63.9934f, -64.0589f,
    -64.0916f, -64.1281f, -64.1837f, -64.2363f, -64.2770f, -64.2934f,
    -64.3225f, -64.3868f, -64.4234f, -64.4573f, -64.4972f, -64.5551f,
    -64.5868f, -64.6233f, -64.6848f, -64.7299f, -64.7550f, -64.8106f,
    -64.8852f, -64.9567f, -65.0360f, -65.0945f, -65.1731f, -65.2444f,
    -65.2855f, -65.3231f, -65.3765f, -65.4467f, -65.5100f, -65.5836f,
    -65.6743f, -65.7611f, -65.8354f, -65.8935f, -65.9761f, -66.0545f,
    -66.1244f, -66.1970f, -66.2874f, -66.3671f, -66.4253f, -66.4988f,
    -66.5822f, -66.6647f, -66.7598f, -66.8443f, -66.9459f, -67.0204f,
    -67.1024f, -67.1904f, -67.2890f, -67.3920f, -67.4990f, -67.6560f,
    -67.7912f, -67.8871f, -67.9999f, -68.1219f, -68.2043f, -68.2933f,
    -68.3826f, -68.4811f, -68.6007f, -68.7119f, -68.8221f, -68.9302f,
    -69.0493f, -69.1750f, -69.2771f, -69.3844f, -69.4990f, -69.6036f,
    -69.7136f, -69.8386f, -69.9523f, -70.0628f, -70.1746f, -70.2902f,
    -70.3978f, -70.4796f, -70.5335f, -70.5189f, -70.4241f, -70.2812f,
    -70.1093f, -70.0510f, -70.1993f, -70.3491f, -70.5081f, -70.7476f,
    -71.0036f, -71.2459f, -71.5072f, -71.7031f, -71.9048f, -72.1479f,
    -72.3301f, -72.6034f, -72.8438f, -73.2156f, -73.5853f, -73.7625f,
    -74.0639f, -74.5064f, -74.9360f, -75.3742f, -75.7763f, -76.1246f,
    -76.4624f, -76.7765f, -77.1317f, -77.5190f, -77.8954f, -78.2606f,
    -78.6848f, -79.1475f, -79.6327f, -80.1251f, -80.6212f, -81.1785f,
    -81.7839f, -82.3960f, -83.0165f, -83.6428f, -84.2644f, -84.9417f,
    -85.6693f, -86.3048f, -86.8885f, -87.5031f, -88.1544f, -88.7786f,
    -90.1860f};

static constexpr float kEmpiricalSpecNormS0[] = {
    4.7844e-03f, 2.3145e-03f, 1.1381e-03f, 1.3410e-03f, 1.6284e-03f,
    1.3601e-03f, 1.0599e-03f, 1.0851e-03f, 1.1904e-03f, 1.1748e-03f,
    1.1158e-03f, 9.9944e-04f, 8.1417e-04f, 6.7053e-04f, 5.5371e-04f,
    4.8805e-04f, 4.4805e-04f, 4.4902e-04f, 4.4367e-04f, 4.1298e-04f,
    3.9242e-04f, 3.8549e-04f, 3.8750e-04f, 3.8644e-04f, 3.6546e-04f,
    3.3962e-04f, 3.1683e-04f, 3.0448e-04f, 3.0708e-04f, 2.9747e-04f,
    2.6569e-04f, 2.4229e-04f, 2.4168e-04f, 2.3999e-04f, 2.2739e-04f,
    2.1208e-04f, 1.8911e-04f, 1.7261e-04f, 1.7518e-04f, 1.7160e-04f,
    1.5553e-04f, 1.4251e-04f, 1.3752e-04f, 1.3577e-04f, 1.3799e-04f,
    1.3612e-04f, 1.2774e-04f, 1.2313e-04f, 1.2493e-04f, 1.2583e-04f,
    1.2150e-04f, 1.1667e-04f, 1.1219e-04f, 1.0793e-04f, 1.0796e-04f,
    1.1019e-04f, 1.0533e-04f, 9.5074e-05f, 9.1160e-05f, 9.1670e-05f,
    8.8261e-05f, 8.5424e-05f, 8.1251e-05f, 7.6537e-05f, 7.5368e-05f,
    7.4621e-05f, 7.3294e-05f, 7.2258e-05f, 7.0843e-05f, 6.9355e-05f,
    6.9455e-05f, 7.0497e-05f, 6.9709e-05f, 6.8114e-05f, 6.9217e-05f,
    6.9024e-05f, 6.7724e-05f, 6.7785e-05f, 6.7715e-05f, 6.6654e-05f,
    6.4153e-05f, 6.2390e-05f, 6.1973e-05f, 6.1699e-05f, 6.0428e-05f,
    5.8661e-05f, 5.7287e-05f, 5.5862e-05f, 5.5363e-05f, 5.4702e-05f,
    5.2889e-05f, 4.9619e-05f, 4.8314e-05f, 4.7766e-05f, 4.6633e-05f,
    4.5958e-05f};

template <typename SessionLike>
std::vector<int64_t> GetTensorShape(SessionLike *sess, size_t index,
                                    bool is_input) {
  if (is_input) {
    return sess->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
  }

  return sess->GetOutputTypeInfo(index)
      .GetTensorTypeAndShapeInfo()
      .GetShape();
}

void FillLinearState(float start, float end, float *dst, int32_t n) {
  if (n <= 0) {
    return;
  }

  if (n == 1) {
    dst[0] = start;
    return;
  }

  float step = (end - start) / (n - 1);
  for (int32_t i = 0; i != n; ++i) {
    dst[i] = start + i * step;
  }
}

}  // namespace

class OfflineSpeechDenoiserDpdfNetModel::Impl {
 public:
  explicit Impl(const OfflineSpeechDenoiserModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config.dpdfnet.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineSpeechDenoiserModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.dpdfnet.model);
    Init(buf.data(), buf.size());
  }

  Ort::Value GetInitState() {
    Ort::Value state = Ort::Value::CreateTensor<float>(
        allocator_, meta_.state_shape.data(), meta_.state_shape.size());

    Fill<float>(&state, 0);

    auto *p = state.GetTensorMutableData<float>();
    if (meta_.profile == "dpdfnet_16khz") {
      FillLinearState(-60.f, -90.f, p, kDpdfNet16kErbBins);
      FillLinearState(1.0e-3f, 1.0e-4f, p + kDpdfNet16kErbBins,
                      kDpdfNet16kSpecBins);
    } else if (meta_.profile == "dpdfnet2_48khz_hr") {
      std::copy(std::begin(kEmpiricalMagNormMu0),
                std::end(kEmpiricalMagNormMu0), p);
      std::copy(std::begin(kEmpiricalSpecNormS0),
                std::end(kEmpiricalSpecNormS0),
                p + std::size(kEmpiricalMagNormMu0));
    } else {
      SHERPA_ONNX_LOGE("Unsupported DPDFNet profile: %s",
                       meta_.profile.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    return state;
  }

  std::pair<Ort::Value, Ort::Value> Run(Ort::Value x,
                                        Ort::Value state) const {
    std::array<Ort::Value, 2> inputs{std::move(x), std::move(state)};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    return {std::move(out[0]), std::move(out[1])};
  }

  const OfflineSpeechDenoiserDpdfNetModelMetaData &GetMetaData() const {
    return meta_;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    if (input_names_.size() != 2 || output_names_.size() != 2) {
      SHERPA_ONNX_LOGE(
          "Expect the dpdfnet model to have 2 inputs and 2 outputs. "
          "Got %zu inputs and %zu outputs.",
          input_names_.size(), output_names_.size());
      SHERPA_ONNX_EXIT(-1);
    }

    auto spec_shape = GetTensorShape(sess_.get(), 0, true);
    auto state_shape = GetTensorShape(sess_.get(), 1, true);
    auto out_spec_shape = GetTensorShape(sess_.get(), 0, false);
    auto out_state_shape = GetTensorShape(sess_.get(), 1, false);

    if (spec_shape.size() != 4 || state_shape.size() != 1 ||
        out_spec_shape.size() != 4 || out_state_shape.size() != 1) {
      SHERPA_ONNX_LOGE(
          "Unexpected dpdfnet ONNX signature. Expected "
          "(spec:[B,T,F,2], state:[S]) -> (spec_e:[B,T,F,2], state_out:[S]).");
      SHERPA_ONNX_EXIT(-1);
    }

    const int64_t freq_bins = spec_shape[2];
    const int64_t complex_dim = spec_shape[3];
    const int64_t state_size = state_shape[0];

    if (freq_bins <= 1 || complex_dim != 2 || state_size <= 0) {
      SHERPA_ONNX_LOGE(
          "Unsupported dpdfnet model shapes. spec=%zu dims, state=%zu dims",
          spec_shape.size(), state_shape.size());
      SHERPA_ONNX_EXIT(-1);
    }

    meta_.spec_shape = std::move(spec_shape);
    meta_.state_shape = std::move(state_shape);
    meta_.state_size = static_cast<int32_t>(state_size);
    meta_.n_fft = static_cast<int32_t>((freq_bins - 1) * 2);
    meta_.window_length = meta_.n_fft;
    meta_.hop_length = meta_.n_fft / 2;

    int32_t required_prefix_state_size = 0;
    if (freq_bins == kDpdfNet16kFreqBins) {
      meta_.sample_rate = kDpdfNet16kSampleRate;
      meta_.profile = "dpdfnet_16khz";
      required_prefix_state_size = kDpdfNet16kErbBins + kDpdfNet16kSpecBins;
    } else if (freq_bins == kDpdfNet48kFreqBins) {
      meta_.sample_rate = kDpdfNet48kSampleRate;
      meta_.profile = "dpdfnet2_48khz_hr";
      required_prefix_state_size = static_cast<int32_t>(
          std::size(kEmpiricalMagNormMu0) + std::size(kEmpiricalSpecNormS0));
    } else {
      SHERPA_ONNX_LOGE(
          "Unsupported DPDFNet model. Expected %d or %d frequency bins, got "
          "%lld.",
          kDpdfNet16kFreqBins, kDpdfNet48kFreqBins,
          static_cast<long long>(freq_bins));
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta_.state_size < required_prefix_state_size) {
      SHERPA_ONNX_LOGE(
          "The dpdfnet state tensor is too small: %d. It must be at least %zu.",
          meta_.state_size, static_cast<size_t>(required_prefix_state_size));
      SHERPA_ONNX_EXIT(-1);
    }

    if (out_spec_shape[2] != freq_bins || out_spec_shape[3] != 2 ||
        out_state_shape[0] != state_size) {
      SHERPA_ONNX_LOGE("Unexpected dpdfnet output shapes.");
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      std::ostringstream os;
      os << "---dpdfnet model---\n";
      os << "input names:\n";
      for (int32_t i = 0; i != static_cast<int32_t>(input_names_.size()); ++i) {
        os << i << " " << input_names_[i] << "\n";
      }

      os << "output names:\n";
      for (int32_t i = 0; i != static_cast<int32_t>(output_names_.size());
           ++i) {
        os << i << " " << output_names_[i] << "\n";
      }

      os << "spec shape: ";
      for (auto d : meta_.spec_shape) {
        os << d << " ";
      }
      os << "\nstate shape: ";
      for (auto d : meta_.state_shape) {
        os << d << " ";
      }
      os << "\nprofile: " << meta_.profile;
      os << "\nsample_rate: " << meta_.sample_rate;
      os << "\nn_fft: " << meta_.n_fft;
      os << "\n";

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }
  }

 private:
  OfflineSpeechDenoiserModelConfig config_;
  OfflineSpeechDenoiserDpdfNetModelMetaData meta_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineSpeechDenoiserDpdfNetModel::~OfflineSpeechDenoiserDpdfNetModel() =
    default;

OfflineSpeechDenoiserDpdfNetModel::OfflineSpeechDenoiserDpdfNetModel(
    const OfflineSpeechDenoiserModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSpeechDenoiserDpdfNetModel::OfflineSpeechDenoiserDpdfNetModel(
    Manager *mgr, const OfflineSpeechDenoiserModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

Ort::Value OfflineSpeechDenoiserDpdfNetModel::GetInitState() const {
  return impl_->GetInitState();
}

std::pair<Ort::Value, Ort::Value> OfflineSpeechDenoiserDpdfNetModel::Run(
    Ort::Value x, Ort::Value state) const {
  return impl_->Run(std::move(x), std::move(state));
}

const OfflineSpeechDenoiserDpdfNetModelMetaData &
OfflineSpeechDenoiserDpdfNetModel::GetMetaData() const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineSpeechDenoiserDpdfNetModel::OfflineSpeechDenoiserDpdfNetModel(
    AAssetManager *mgr, const OfflineSpeechDenoiserModelConfig &config);
#endif

#if __OHOS__
template OfflineSpeechDenoiserDpdfNetModel::OfflineSpeechDenoiserDpdfNetModel(
    NativeResourceManager *mgr, const OfflineSpeechDenoiserModelConfig &config);
#endif

}  // namespace sherpa_onnx
