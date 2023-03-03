// sherpa-onnx/c-api/c-api.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/c-api/c-api.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/online-recognizer.h"

struct SherpaOnnxOnlineRecognizer {
  sherpa_onnx::OnlineRecognizer *impl;
};

struct SherpaOnnxOnlineStream {
  std::unique_ptr<sherpa_onnx::OnlineStream> impl;
  explicit SherpaOnnxOnlineStream(std::unique_ptr<sherpa_onnx::OnlineStream> p)
      : impl(std::move(p)) {}
};

struct SherpaOnnxDisplay {
  std::unique_ptr<sherpa_onnx::Display> impl;
};

SherpaOnnxOnlineRecognizer *CreateOnlineRecognizer(
    const SherpaOnnxOnlineRecognizerConfig *config) {
  sherpa_onnx::OnlineRecognizerConfig recognizer_config;

  recognizer_config.feat_config.sampling_rate = config->feat_config.sample_rate;
  recognizer_config.feat_config.feature_dim = config->feat_config.feature_dim;

  recognizer_config.model_config.encoder_filename =
      config->model_config.encoder;
  recognizer_config.model_config.decoder_filename =
      config->model_config.decoder;
  recognizer_config.model_config.joiner_filename = config->model_config.joiner;
  recognizer_config.model_config.tokens = config->model_config.tokens;
  recognizer_config.model_config.num_threads = config->model_config.num_threads;
  recognizer_config.model_config.debug = config->model_config.debug;

  recognizer_config.decoding_method = config->decoding_method;
  recognizer_config.max_active_paths = config->max_active_paths;

  recognizer_config.enable_endpoint = config->enable_endpoint;

  recognizer_config.endpoint_config.rule1.min_trailing_silence =
      config->rule1_min_trailing_silence;

  recognizer_config.endpoint_config.rule2.min_trailing_silence =
      config->rule2_min_trailing_silence;

  recognizer_config.endpoint_config.rule3.min_utterance_length =
      config->rule3_min_utterance_length;

  SherpaOnnxOnlineRecognizer *recognizer = new SherpaOnnxOnlineRecognizer;
  recognizer->impl = new sherpa_onnx::OnlineRecognizer(recognizer_config);

  return recognizer;
}

void DestroyOnlineRecognizer(SherpaOnnxOnlineRecognizer *recognizer) {
  delete recognizer->impl;
  delete recognizer;
}

SherpaOnnxOnlineStream *CreateOnlineStream(
    const SherpaOnnxOnlineRecognizer *recognizer) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(recognizer->impl->CreateStream());
  return stream;
}

void DestoryOnlineStream(SherpaOnnxOnlineStream *stream) { delete stream; }

void AcceptWaveform(SherpaOnnxOnlineStream *stream, int32_t sample_rate,
                    const float *samples, int32_t n) {
  stream->impl->AcceptWaveform(sample_rate, samples, n);
}

int32_t IsOnlineStreamReady(SherpaOnnxOnlineRecognizer *recognizer,
                            SherpaOnnxOnlineStream *stream) {
  return recognizer->impl->IsReady(stream->impl.get());
}

void DecodeOnlineStream(SherpaOnnxOnlineRecognizer *recognizer,
                        SherpaOnnxOnlineStream *stream) {
  recognizer->impl->DecodeStream(stream->impl.get());
}

void DecodeMultipleOnlineStreams(SherpaOnnxOnlineRecognizer *recognizer,
                                 SherpaOnnxOnlineStream **streams, int32_t n) {
  std::vector<sherpa_onnx::OnlineStream *> ss(n);
  for (int32_t i = 0; i != n; ++n) {
    ss[i] = streams[i]->impl.get();
  }
  recognizer->impl->DecodeStreams(ss.data(), n);
}

SherpaOnnxOnlineRecognizerResult *GetOnlineStreamResult(
    SherpaOnnxOnlineRecognizer *recognizer, SherpaOnnxOnlineStream *stream) {
  sherpa_onnx::OnlineRecognizerResult result =
      recognizer->impl->GetResult(stream->impl.get());
  const auto &text = result.text;

  auto r = new SherpaOnnxOnlineRecognizerResult;
  r->text = new char[text.size() + 1];
  std::copy(text.begin(), text.end(), const_cast<char *>(r->text));
  const_cast<char *>(r->text)[text.size()] = 0;

  return r;
}

void DestroyOnlineRecognizerResult(const SherpaOnnxOnlineRecognizerResult *r) {
  delete[] r->text;
  delete r;
}

void Reset(SherpaOnnxOnlineRecognizer *recognizer,
           SherpaOnnxOnlineStream *stream) {
  recognizer->impl->Reset(stream->impl.get());
}

void InputFinished(SherpaOnnxOnlineStream *stream) {
  stream->impl->InputFinished();
}

int32_t IsEndpoint(SherpaOnnxOnlineRecognizer *recognizer,
                   SherpaOnnxOnlineStream *stream) {
  return recognizer->impl->IsEndpoint(stream->impl.get());
}

SherpaOnnxDisplay *CreateDisplay(int32_t max_word_per_line) {
  SherpaOnnxDisplay *ans = new SherpaOnnxDisplay;
  ans->impl = std::make_unique<sherpa_onnx::Display>(max_word_per_line);
  return ans;
}

void DestroyDisplay(SherpaOnnxDisplay *display) { delete display; }

void SherpaOnnxPrint(SherpaOnnxDisplay *display, int32_t idx, const char *s) {
  display->impl->Print(idx, s);
}
