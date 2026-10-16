// Copyright (c) 2026 LittleMouse
#include "sherpa-onnx/csrc/axera/offline-tts-zipvoice-model-axera.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/axera/tts-session.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/normal-data-generator.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-length.h"

namespace sherpa_onnx {
class OfflineTtsZipvoiceModelAxera::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config) {
    meta_.max_tokens = 384;
    meta_.max_frames = 1024;
    meta_.max_generated_frames = 620;
    auto bytes = ReadFile(config.zipvoice.decoder);
    auto manifest = nlohmann::json::parse(bytes.begin(), bytes.end());
    int32_t version = manifest.at("version").get<int32_t>();
    if ((version != 1 && version != 2) ||
        manifest.at("decoder_parts").size() != 4) {
      throw std::runtime_error(
          "Expected an AX650 four-part decoder manifest (version 1 or 2)");
    }
    standard_ = version == 2;
    int32_t batch = standard_ ? 2 : 1;
    std::string dir = config.zipvoice.decoder.substr(
        0, config.zipvoice.decoder.find_last_of("/\\") + 1);
    encoder_ = std::make_unique<AxeraTtsSession>(config.zipvoice.encoder,
                                                 config.debug);
    encoder_->CheckCount(1, 1);
    encoder_->CheckInput("cat_tokens", AX_ENGINE_DT_SINT32, {1, 384});
    encoder_->CheckOutput("encoded", AX_ENGINE_DT_FLOAT32, {1, 384, 100});
    for (int32_t i = 0; i < 4; ++i) {
      auto file =
          manifest.at("decoder_parts").at(i).at("file").get<std::string>();
      auto part = std::make_unique<AxeraTtsSession>(dir + file, config.debug);
      part->CheckCount(i == 0 ? 6 : (standard_ && i == 3 ? 4 : 3),
                       i == 0 ? (standard_ ? 4 : 2) : 1);
      part->CheckInput(standard_ && i > 0 ? "padding_mask2" : "padding_mask",
                       AX_ENGINE_DT_UINT8, {i == 0 ? 1 : batch, 1024});
      if (i == 0) {
        part->CheckInput("t", AX_ENGINE_DT_FLOAT32, {1});
        part->CheckInput("guidance_scale", AX_ENGINE_DT_FLOAT32, {1});
        for (const auto *name : {"x", "text_condition", "speech_condition"}) {
          part->CheckInput(name, AX_ENGINE_DT_FLOAT32, {1, 1024, 100});
        }
        part->CheckOutput("time_hidden", AX_ENGINE_DT_FLOAT32, {batch, 192});
        if (standard_) {
          part->CheckOutput("padding_mask2", AX_ENGINE_DT_UINT8, {2, 1024});
          part->CheckOutput("cfg_scale", AX_ENGINE_DT_FLOAT32, {1});
        }
      } else {
        part->CheckInput("decoder_hidden_p" + std::to_string(i - 1),
                         AX_ENGINE_DT_FLOAT32, {1024, batch, 512});
        part->CheckInput("time_hidden", AX_ENGINE_DT_FLOAT32, {batch, 192});
      }
      if (standard_ && i == 3) {
        part->CheckInput("cfg_scale", AX_ENGINE_DT_FLOAT32, {1});
      }
      if (i == 3)
        part->CheckOutput("v", AX_ENGINE_DT_FLOAT32, {1, 1024, 100});
      else
        part->CheckOutput("decoder_hidden_p" + std::to_string(i),
                          AX_ENGINE_DT_FLOAT32, {1024, batch, 512});
      decoder_.push_back(std::move(part));
    }
  }
  Ort::Value Run(Ort::Value tokens, Ort::Value prompt_tokens,
                 Ort::Value prompt_features, float speed, int32_t steps,
                 float shift, float guidance) const {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
      auto p = prompt_tokens.GetTensorTypeAndShapeInfo().GetElementCount();
      auto n = tokens.GetTensorTypeAndShapeInfo().GetElementCount();
      auto shape = prompt_features.GetTensorTypeAndShapeInfo().GetShape();
      if (p >= 384 || n >= 384 || shape.size() != 3 || shape[0] != 1 ||
          shape[2] != 100 || shape[1] <= 0 || shape[1] >= 1024 || steps <= 0 ||
          !std::isfinite(shift) || shift <= 0 || !std::isfinite(guidance) ||
          guidance <= 0) {
        throw std::runtime_error(
            "Invalid AXERA ZipVoice input or sampling parameters");
      }
      int32_t r = shape[1];
      int32_t f = ZipvoiceStaticFeatureLength(p, n, r, speed, 384, 1024, 620);
      if (!f || f < p + n)
        throw std::runtime_error(
            "AXERA ZipVoice chunk exceeds static capacity");
      std::vector<int32_t> ids(384, 0);
      const auto *prompt = prompt_tokens.GetTensorData<int64_t>();
      const auto *text = tokens.GetTensorData<int64_t>();
      for (size_t i = 0; i < p + n; ++i) {
        int64_t id = i < p ? prompt[i] : text[i - p];
        if (id < 0 || id >= 360)
          throw std::runtime_error("Invalid AXERA token ID");
        ids[i] = id;
      }
      encoder_->Set("cat_tokens", ids.data(), ids.size() * sizeof(int32_t));
      encoder_->Run();
      auto encoded = encoder_->Get("encoded");
      std::vector<float> condition(1024 * 100, 0), speech(condition.size(), 0),
          x(condition.size(), 0);
      int32_t duration = f / (p + n);
      for (int32_t frame = 0; frame < f; ++frame) {
        // Match the static export's repeat + residual pad embedding expansion.
        int32_t token = std::min<int32_t>(frame / duration, p + n);
        std::copy_n(encoded.data() + token * 100, 100,
                    condition.data() + frame * 100);
      }
      std::copy_n(prompt_features.GetTensorData<float>(), r * 100,
                  speech.data());
      normal_.Fill(x.data(), f * 100);
      std::vector<uint8_t> mask(1024, 1);
      std::fill_n(mask.begin(), f, 0);
      auto timestep = [shift, steps](int32_t i) {
        float t = static_cast<float>(i) / steps;
        return shift * t / (1 + (shift - 1) * t);
      };
      for (int32_t step = 0; step < steps; ++step) {
        float t = timestep(step);
        decoder_[0]->Set("t", &t, sizeof(t));
        decoder_[0]->Set("guidance_scale", &guidance, sizeof(guidance));
        decoder_[0]->Set("x", x.data(), x.size() * sizeof(float));
        decoder_[0]->Set("text_condition", condition.data(),
                         condition.size() * sizeof(float));
        decoder_[0]->Set("speech_condition", speech.data(),
                         speech.size() * sizeof(float));
        decoder_[0]->Set("padding_mask", mask.data(), mask.size());
        decoder_[0]->Run();
        auto hidden = decoder_[0]->Get("decoder_hidden_p0");
        auto time = decoder_[0]->Get("time_hidden");
        auto partition_mask =
            standard_ ? decoder_[0]->GetBytes("padding_mask2") : mask;
        std::vector<float> cfg_scale;
        if (standard_) cfg_scale = decoder_[0]->Get("cfg_scale");
        for (int32_t i = 1; i < 4; ++i) {
          decoder_[i]->Set("decoder_hidden_p" + std::to_string(i - 1),
                           hidden.data(), hidden.size() * sizeof(float));
          decoder_[i]->Set("time_hidden", time.data(),
                           time.size() * sizeof(float));
          decoder_[i]->Set(standard_ ? "padding_mask2" : "padding_mask",
                           partition_mask.data(), partition_mask.size());
          if (standard_ && i == 3) {
            decoder_[i]->Set("cfg_scale", cfg_scale.data(), sizeof(float));
          }
          decoder_[i]->Run();
          hidden = decoder_[i]->Get(
              i == 3 ? "v" : "decoder_hidden_p" + std::to_string(i));
        }
        float dt = timestep(step + 1) - t;
        for (int32_t i = 0; i < f * 100; ++i) x[i] += hidden[i] * dt;
      }
      std::vector<int64_t> out_shape{1, f - r, 100};
      Ort::AllocatorWithDefaultOptions allocator;
      auto out = Ort::Value::CreateTensor<float>(allocator, out_shape.data(),
                                                 out_shape.size());
      std::copy(x.begin() + r * 100, x.begin() + f * 100,
                out.GetTensorMutableData<float>());
      return out;
    } catch (const std::exception &e) {
      SHERPA_ONNX_LOGE("AXERA ZipVoice generation failed: %s", e.what());
      return Ort::Value{nullptr};
    }
  }
  OfflineTtsZipvoiceModelMetaData meta_;

 private:
  std::unique_ptr<AxeraTtsSession> encoder_;
  std::vector<std::unique_ptr<AxeraTtsSession>> decoder_;
  mutable std::mutex mutex_;
  NormalDataGenerator normal_;
  bool standard_ = false;
};
OfflineTtsZipvoiceModelAxera::OfflineTtsZipvoiceModelAxera(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}
OfflineTtsZipvoiceModelAxera::~OfflineTtsZipvoiceModelAxera() = default;
const OfflineTtsZipvoiceModelMetaData &
OfflineTtsZipvoiceModelAxera::GetMetaData() const {
  return impl_->meta_;
}
Ort::Value OfflineTtsZipvoiceModelAxera::Run(
    Ort::Value tokens, Ort::Value prompt_tokens, Ort::Value prompt_features,
    float speed, int32_t num_steps, float t_shift, float guidance_scale) const {
  return impl_->Run(std::move(tokens), std::move(prompt_tokens),
                    std::move(prompt_features), speed, num_steps, t_shift,
                    guidance_scale);
}
}  // namespace sherpa_onnx
