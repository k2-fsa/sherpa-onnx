// sherpa-onnx/c-api/cxx-api.h
//
// Copyright (c)  2024  Xiaomi Corporation

// C++ Wrapper of the C API for sherpa-onnx
#ifndef SHERPA_ONNX_C_API_CXX_API_H_
#define SHERPA_ONNX_C_API_CXX_API_H_

#include <string>
#include <vector>

#include "sherpa-onnx/c-api/c-api.h"

namespace sherpa_onnx::cxx {

struct SHERPA_ONNX_API OnlineTransducerModelConfig {
  std::string encoder;
  std::string decoder;
  std::string joiner;
};

struct SHERPA_ONNX_API OnlineParaformerModelConfig {
  std::string encoder;
  std::string decoder;
};

struct SHERPA_ONNX_API OnlineZipformer2CtcModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OnlineModelConfig {
  OnlineTransducerModelConfig transducer;
  OnlineParaformerModelConfig paraformer;
  OnlineZipformer2CtcModelConfig zipformer2_ctc;
  std::string tokens;
  int32_t num_threads = 1;
  std::string provider = "cpu";
  bool debug = false;
  std::string model_type;
  std::string modeling_unit = "cjkchar";
  std::string bpe_vocab;
  std::string tokens_buf;
};

struct SHERPA_ONNX_API FeatureConfig {
  int32_t sample_rate = 16000;
  int32_t feature_dim = 80;
};

struct SHERPA_ONNX_API OnlineCtcFstDecoderConfig {
  std::string graph;
  int32_t max_active = 3000;
};

struct SHERPA_ONNX_API OnlineRecognizerConfig {
  FeatureConfig feat_config;
  OnlineModelConfig model_config;

  std::string decoding_method = "greedy_search";

  int32_t max_active_paths = 4;

  bool enable_endpoint = false;

  float rule1_min_trailing_silence = 2.4;

  float rule2_min_trailing_silence = 1.2;

  float rule3_min_utterance_length = 20;

  std::string hotwords_file;

  float hotwords_score = 1.5;

  OnlineCtcFstDecoderConfig ctc_fst_decoder_config;
  std::string rule_fsts;
  std::string rule_fars;
  float blank_penalty = 0;

  std::string hotwords_buf;
};

struct SHERPA_ONNX_API OnlineRecognizerResult {
  std::string text;
  std::vector<std::string> tokens;
  std::vector<float> timestamps;
  std::string json;
};

struct SHERPA_ONNX_API Wave {
  std::vector<float> samples;
  int32_t sample_rate;
};

SHERPA_ONNX_API Wave ReadWave(const std::string &filename);

template <typename Derived, typename T>
class SHERPA_ONNX_API MoveOnly {
 public:
  explicit MoveOnly(const T *p) : p_(p) {}

  ~MoveOnly() { Destroy(); }

  MoveOnly(const MoveOnly &) = delete;

  MoveOnly &operator=(const MoveOnly &) = delete;

  MoveOnly(MoveOnly &&other) : p_(other.Release()) {}

  MoveOnly &operator=(MoveOnly &&other) {
    if (&other == this) {
      return *this;
    }

    Destroy();

    p_ = other.Release();
  }

  const T *Get() const { return p_; }

  const T *Release() {
    const T *p = p_;
    p_ = nullptr;
    return p;
  }

 private:
  void Destroy() {
    if (p_ == nullptr) {
      return;
    }

    static_cast<Derived *>(this)->Destroy(p_);

    p_ = nullptr;
  }

 protected:
  const T *p_ = nullptr;
};

class SHERPA_ONNX_API OnlineStream
    : public MoveOnly<OnlineStream, SherpaOnnxOnlineStream> {
 public:
  explicit OnlineStream(const SherpaOnnxOnlineStream *p);

  void AcceptWaveform(int32_t sample_rate, const float *samples,
                      int32_t n) const;

  void Destroy(const SherpaOnnxOnlineStream *p) const;
};

class SHERPA_ONNX_API OnlineRecognizer
    : public MoveOnly<OnlineRecognizer, SherpaOnnxOnlineRecognizer> {
 public:
  static OnlineRecognizer Create(const OnlineRecognizerConfig &config);

  void Destroy(const SherpaOnnxOnlineRecognizer *p) const;

  OnlineStream CreateStream() const;

  OnlineStream CreateStream(const std::string &hotwords) const;

  bool IsReady(const OnlineStream *s) const;

  void Decode(const OnlineStream *s) const;

  void Decode(const OnlineStream *ss, int32_t n) const;

  OnlineRecognizerResult GetResult(const OnlineStream *s) const;

 private:
  explicit OnlineRecognizer(const SherpaOnnxOnlineRecognizer *p);
};

}  // namespace sherpa_onnx::cxx

#endif  // SHERPA_ONNX_C_API_CXX_API_H_
