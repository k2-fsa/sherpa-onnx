// sherpa-onnx/c-api/c-api.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/c-api/c-api.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"
#include "sherpa-onnx/csrc/wave-writer.h"

struct SherpaOnnxOnlineRecognizer {
  std::unique_ptr<sherpa_onnx::OnlineRecognizer> impl;
};

struct SherpaOnnxOnlineStream {
  std::unique_ptr<sherpa_onnx::OnlineStream> impl;
  explicit SherpaOnnxOnlineStream(std::unique_ptr<sherpa_onnx::OnlineStream> p)
      : impl(std::move(p)) {}
};

struct SherpaOnnxDisplay {
  std::unique_ptr<sherpa_onnx::Display> impl;
};

#define SHERPA_ONNX_OR(x, y) (x ? x : y)

SherpaOnnxOnlineRecognizer *CreateOnlineRecognizer(
    const SherpaOnnxOnlineRecognizerConfig *config) {
  sherpa_onnx::OnlineRecognizerConfig recognizer_config;

  recognizer_config.feat_config.sampling_rate =
      SHERPA_ONNX_OR(config->feat_config.sample_rate, 16000);
  recognizer_config.feat_config.feature_dim =
      SHERPA_ONNX_OR(config->feat_config.feature_dim, 80);

  recognizer_config.model_config.transducer.encoder =
      SHERPA_ONNX_OR(config->model_config.transducer.encoder, "");
  recognizer_config.model_config.transducer.decoder =
      SHERPA_ONNX_OR(config->model_config.transducer.decoder, "");
  recognizer_config.model_config.transducer.joiner =
      SHERPA_ONNX_OR(config->model_config.transducer.joiner, "");

  recognizer_config.model_config.paraformer.encoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.encoder, "");
  recognizer_config.model_config.paraformer.decoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.decoder, "");

  recognizer_config.model_config.zipformer2_ctc.model =
      SHERPA_ONNX_OR(config->model_config.zipformer2_ctc.model, "");

  recognizer_config.model_config.tokens =
      SHERPA_ONNX_OR(config->model_config.tokens, "");
  recognizer_config.model_config.num_threads =
      SHERPA_ONNX_OR(config->model_config.num_threads, 1);
  recognizer_config.model_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");
  recognizer_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");
  recognizer_config.model_config.debug =
      SHERPA_ONNX_OR(config->model_config.debug, 0);

  recognizer_config.decoding_method =
      SHERPA_ONNX_OR(config->decoding_method, "greedy_search");
  recognizer_config.max_active_paths =
      SHERPA_ONNX_OR(config->max_active_paths, 4);

  recognizer_config.enable_endpoint =
      SHERPA_ONNX_OR(config->enable_endpoint, 0);

  recognizer_config.endpoint_config.rule1.min_trailing_silence =
      SHERPA_ONNX_OR(config->rule1_min_trailing_silence, 2.4);

  recognizer_config.endpoint_config.rule2.min_trailing_silence =
      SHERPA_ONNX_OR(config->rule2_min_trailing_silence, 1.2);

  recognizer_config.endpoint_config.rule3.min_utterance_length =
      SHERPA_ONNX_OR(config->rule3_min_utterance_length, 20);

  recognizer_config.hotwords_file = SHERPA_ONNX_OR(config->hotwords_file, "");
  recognizer_config.hotwords_score =
      SHERPA_ONNX_OR(config->hotwords_score, 1.5);

  if (config->model_config.debug) {
    fprintf(stderr, "%s\n", recognizer_config.ToString().c_str());
  }

  SherpaOnnxOnlineRecognizer *recognizer = new SherpaOnnxOnlineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_onnx::OnlineRecognizer>(recognizer_config);

  return recognizer;
}

void DestroyOnlineRecognizer(SherpaOnnxOnlineRecognizer *recognizer) {
  delete recognizer;
}

SherpaOnnxOnlineStream *CreateOnlineStream(
    const SherpaOnnxOnlineRecognizer *recognizer) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(recognizer->impl->CreateStream());
  return stream;
}

SherpaOnnxOnlineStream *CreateOnlineStreamWithHotwords(
    const SherpaOnnxOnlineRecognizer *recognizer, const char *hotwords) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(recognizer->impl->CreateStream(hotwords));
  return stream;
}

void DestroyOnlineStream(SherpaOnnxOnlineStream *stream) { delete stream; }

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
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  recognizer->impl->DecodeStreams(ss.data(), n);
}

const SherpaOnnxOnlineRecognizerResult *GetOnlineStreamResult(
    SherpaOnnxOnlineRecognizer *recognizer, SherpaOnnxOnlineStream *stream) {
  sherpa_onnx::OnlineRecognizerResult result =
      recognizer->impl->GetResult(stream->impl.get());
  const auto &text = result.text;

  auto r = new SherpaOnnxOnlineRecognizerResult;
  memset(r, 0, sizeof(SherpaOnnxOnlineRecognizerResult));

  // copy text
  r->text = new char[text.size() + 1];
  std::copy(text.begin(), text.end(), const_cast<char *>(r->text));
  const_cast<char *>(r->text)[text.size()] = 0;

  // copy json
  const auto &json = result.AsJsonString();
  r->json = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), const_cast<char *>(r->json));
  const_cast<char *>(r->json)[json.size()] = 0;

  // copy tokens
  auto count = result.tokens.size();
  if (count > 0) {
    size_t total_length = 0;
    for (const auto &token : result.tokens) {
      // +1 for the null character at the end of each token
      total_length += token.size() + 1;
    }

    r->count = count;
    // Each word ends with nullptr
    r->tokens = new char[total_length];
    memset(reinterpret_cast<void *>(const_cast<char *>(r->tokens)), 0,
           total_length);
    char **tokens_temp = new char *[r->count];
    int32_t pos = 0;
    for (int32_t i = 0; i < r->count; ++i) {
      tokens_temp[i] = const_cast<char *>(r->tokens) + pos;
      memcpy(reinterpret_cast<void *>(const_cast<char *>(r->tokens + pos)),
             result.tokens[i].c_str(), result.tokens[i].size());
      // +1 to move past the null character
      pos += result.tokens[i].size() + 1;
    }
    r->tokens_arr = tokens_temp;

    if (!result.timestamps.empty()) {
      r->timestamps = new float[r->count];
      std::copy(result.timestamps.begin(), result.timestamps.end(),
                r->timestamps);
    } else {
      r->timestamps = nullptr;
    }

  } else {
    r->count = 0;
    r->timestamps = nullptr;
    r->tokens = nullptr;
    r->tokens_arr = nullptr;
  }

  return r;
}

void DestroyOnlineRecognizerResult(const SherpaOnnxOnlineRecognizerResult *r) {
  if (r) {
    delete[] r->text;
    delete[] r->json;
    delete[] r->tokens;
    delete[] r->tokens_arr;
    delete[] r->timestamps;
    delete r;
  }
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

// ============================================================
// For offline ASR (i.e., non-streaming ASR)
// ============================================================
//
struct SherpaOnnxOfflineRecognizer {
  std::unique_ptr<sherpa_onnx::OfflineRecognizer> impl;
};

struct SherpaOnnxOfflineStream {
  std::unique_ptr<sherpa_onnx::OfflineStream> impl;
  explicit SherpaOnnxOfflineStream(
      std::unique_ptr<sherpa_onnx::OfflineStream> p)
      : impl(std::move(p)) {}
};

SherpaOnnxOfflineRecognizer *CreateOfflineRecognizer(
    const SherpaOnnxOfflineRecognizerConfig *config) {
  sherpa_onnx::OfflineRecognizerConfig recognizer_config;

  recognizer_config.feat_config.sampling_rate =
      SHERPA_ONNX_OR(config->feat_config.sample_rate, 16000);

  recognizer_config.feat_config.feature_dim =
      SHERPA_ONNX_OR(config->feat_config.feature_dim, 80);

  recognizer_config.model_config.transducer.encoder_filename =
      SHERPA_ONNX_OR(config->model_config.transducer.encoder, "");

  recognizer_config.model_config.transducer.decoder_filename =
      SHERPA_ONNX_OR(config->model_config.transducer.decoder, "");

  recognizer_config.model_config.transducer.joiner_filename =
      SHERPA_ONNX_OR(config->model_config.transducer.joiner, "");

  recognizer_config.model_config.paraformer.model =
      SHERPA_ONNX_OR(config->model_config.paraformer.model, "");

  recognizer_config.model_config.nemo_ctc.model =
      SHERPA_ONNX_OR(config->model_config.nemo_ctc.model, "");

  recognizer_config.model_config.whisper.encoder =
      SHERPA_ONNX_OR(config->model_config.whisper.encoder, "");

  recognizer_config.model_config.whisper.decoder =
      SHERPA_ONNX_OR(config->model_config.whisper.decoder, "");

  recognizer_config.model_config.tdnn.model =
      SHERPA_ONNX_OR(config->model_config.tdnn.model, "");

  recognizer_config.model_config.tokens =
      SHERPA_ONNX_OR(config->model_config.tokens, "");
  recognizer_config.model_config.num_threads =
      SHERPA_ONNX_OR(config->model_config.num_threads, 1);
  recognizer_config.model_config.debug =
      SHERPA_ONNX_OR(config->model_config.debug, 0);
  recognizer_config.model_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");
  recognizer_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");

  recognizer_config.lm_config.model =
      SHERPA_ONNX_OR(config->lm_config.model, "");
  recognizer_config.lm_config.scale =
      SHERPA_ONNX_OR(config->lm_config.scale, 1.0);

  recognizer_config.decoding_method =
      SHERPA_ONNX_OR(config->decoding_method, "greedy_search");
  recognizer_config.max_active_paths =
      SHERPA_ONNX_OR(config->max_active_paths, 4);

  recognizer_config.hotwords_file = SHERPA_ONNX_OR(config->hotwords_file, "");
  recognizer_config.hotwords_score =
      SHERPA_ONNX_OR(config->hotwords_score, 1.5);

  if (config->model_config.debug) {
    fprintf(stderr, "%s\n", recognizer_config.ToString().c_str());
  }

  SherpaOnnxOfflineRecognizer *recognizer = new SherpaOnnxOfflineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_onnx::OfflineRecognizer>(recognizer_config);

  return recognizer;
}

void DestroyOfflineRecognizer(SherpaOnnxOfflineRecognizer *recognizer) {
  delete recognizer;
}

SherpaOnnxOfflineStream *CreateOfflineStream(
    const SherpaOnnxOfflineRecognizer *recognizer) {
  SherpaOnnxOfflineStream *stream =
      new SherpaOnnxOfflineStream(recognizer->impl->CreateStream());
  return stream;
}

void DestroyOfflineStream(SherpaOnnxOfflineStream *stream) { delete stream; }

void AcceptWaveformOffline(SherpaOnnxOfflineStream *stream, int32_t sample_rate,
                           const float *samples, int32_t n) {
  stream->impl->AcceptWaveform(sample_rate, samples, n);
}

void DecodeOfflineStream(SherpaOnnxOfflineRecognizer *recognizer,
                         SherpaOnnxOfflineStream *stream) {
  recognizer->impl->DecodeStream(stream->impl.get());
}

void DecodeMultipleOfflineStreams(SherpaOnnxOfflineRecognizer *recognizer,
                                  SherpaOnnxOfflineStream **streams,
                                  int32_t n) {
  std::vector<sherpa_onnx::OfflineStream *> ss(n);
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  recognizer->impl->DecodeStreams(ss.data(), n);
}

const SherpaOnnxOfflineRecognizerResult *GetOfflineStreamResult(
    SherpaOnnxOfflineStream *stream) {
  const sherpa_onnx::OfflineRecognitionResult &result =
      stream->impl->GetResult();
  const auto &text = result.text;

  auto r = new SherpaOnnxOfflineRecognizerResult;
  memset(r, 0, sizeof(SherpaOnnxOfflineRecognizerResult));

  r->text = new char[text.size() + 1];
  std::copy(text.begin(), text.end(), const_cast<char *>(r->text));
  const_cast<char *>(r->text)[text.size()] = 0;

  if (!result.timestamps.empty()) {
    r->timestamps = new float[result.timestamps.size()];
    std::copy(result.timestamps.begin(), result.timestamps.end(),
              r->timestamps);
    r->count = result.timestamps.size();
  } else {
    r->timestamps = nullptr;
    r->count = 0;
  }

  return r;
}

void DestroyOfflineRecognizerResult(
    const SherpaOnnxOfflineRecognizerResult *r) {
  if (r) {
    delete[] r->text;
    delete[] r->timestamps;
    delete r;
  }
}

// ============================================================
// For VAD
// ============================================================
//
struct SherpaOnnxCircularBuffer {
  std::unique_ptr<sherpa_onnx::CircularBuffer> impl;
};

SherpaOnnxCircularBuffer *SherpaOnnxCreateCircularBuffer(int32_t capacity) {
  SherpaOnnxCircularBuffer *buffer = new SherpaOnnxCircularBuffer;
  buffer->impl = std::make_unique<sherpa_onnx::CircularBuffer>(capacity);
  return buffer;
}

void SherpaOnnxDestroyCircularBuffer(SherpaOnnxCircularBuffer *buffer) {
  delete buffer;
}

void SherpaOnnxCircularBufferPush(SherpaOnnxCircularBuffer *buffer,
                                  const float *p, int32_t n) {
  buffer->impl->Push(p, n);
}

const float *SherpaOnnxCircularBufferGet(SherpaOnnxCircularBuffer *buffer,
                                         int32_t start_index, int32_t n) {
  std::vector<float> v = buffer->impl->Get(start_index, n);

  float *p = new float[n];
  std::copy(v.begin(), v.end(), p);
  return p;
}

void SherpaOnnxCircularBufferFree(const float *p) { delete[] p; }

void SherpaOnnxCircularBufferPop(SherpaOnnxCircularBuffer *buffer, int32_t n) {
  buffer->impl->Pop(n);
}

int32_t SherpaOnnxCircularBufferSize(SherpaOnnxCircularBuffer *buffer) {
  return buffer->impl->Size();
}

int32_t SherpaOnnxCircularBufferHead(SherpaOnnxCircularBuffer *buffer) {
  return buffer->impl->Head();
}

void SherpaOnnxCircularBufferReset(SherpaOnnxCircularBuffer *buffer) {
  buffer->impl->Reset();
}

struct SherpaOnnxVoiceActivityDetector {
  std::unique_ptr<sherpa_onnx::VoiceActivityDetector> impl;
};

SherpaOnnxVoiceActivityDetector *SherpaOnnxCreateVoiceActivityDetector(
    const SherpaOnnxVadModelConfig *config, float buffer_size_in_seconds) {
  sherpa_onnx::VadModelConfig vad_config;

  vad_config.silero_vad.model = SHERPA_ONNX_OR(config->silero_vad.model, "");
  vad_config.silero_vad.threshold =
      SHERPA_ONNX_OR(config->silero_vad.threshold, 0.5);

  vad_config.silero_vad.min_silence_duration =
      SHERPA_ONNX_OR(config->silero_vad.min_silence_duration, 0.5);

  vad_config.silero_vad.min_speech_duration =
      SHERPA_ONNX_OR(config->silero_vad.min_speech_duration, 0.25);

  vad_config.silero_vad.window_size =
      SHERPA_ONNX_OR(config->silero_vad.window_size, 512);

  vad_config.sample_rate = SHERPA_ONNX_OR(config->sample_rate, 16000);
  vad_config.num_threads = SHERPA_ONNX_OR(config->num_threads, 1);
  vad_config.provider = SHERPA_ONNX_OR(config->provider, "cpu");
  vad_config.debug = SHERPA_ONNX_OR(config->debug, false);

  if (vad_config.debug) {
    fprintf(stderr, "%s\n", vad_config.ToString().c_str());
  }

  SherpaOnnxVoiceActivityDetector *p = new SherpaOnnxVoiceActivityDetector;
  p->impl = std::make_unique<sherpa_onnx::VoiceActivityDetector>(
      vad_config, buffer_size_in_seconds);

  return p;
}

void SherpaOnnxDestroyVoiceActivityDetector(
    SherpaOnnxVoiceActivityDetector *p) {
  delete p;
}

void SherpaOnnxVoiceActivityDetectorAcceptWaveform(
    SherpaOnnxVoiceActivityDetector *p, const float *samples, int32_t n) {
  p->impl->AcceptWaveform(samples, n);
}

int32_t SherpaOnnxVoiceActivityDetectorEmpty(
    SherpaOnnxVoiceActivityDetector *p) {
  return p->impl->Empty();
}

int32_t SherpaOnnxVoiceActivityDetectorDetected(
    SherpaOnnxVoiceActivityDetector *p) {
  return p->impl->IsSpeechDetected();
}

void SherpaOnnxVoiceActivityDetectorPop(SherpaOnnxVoiceActivityDetector *p) {
  p->impl->Pop();
}

void SherpaOnnxVoiceActivityDetectorClear(SherpaOnnxVoiceActivityDetector *p) {
  p->impl->Clear();
}

const SherpaOnnxSpeechSegment *SherpaOnnxVoiceActivityDetectorFront(
    SherpaOnnxVoiceActivityDetector *p) {
  const sherpa_onnx::SpeechSegment &segment = p->impl->Front();

  SherpaOnnxSpeechSegment *ans = new SherpaOnnxSpeechSegment;
  ans->start = segment.start;
  ans->samples = new float[segment.samples.size()];
  std::copy(segment.samples.begin(), segment.samples.end(), ans->samples);
  ans->n = segment.samples.size();

  return ans;
}

void SherpaOnnxDestroySpeechSegment(const SherpaOnnxSpeechSegment *p) {
  if (p) {
    delete[] p->samples;
    delete p;
  }
}

void SherpaOnnxVoiceActivityDetectorReset(SherpaOnnxVoiceActivityDetector *p) {
  p->impl->Reset();
}

struct SherpaOnnxOfflineTts {
  std::unique_ptr<sherpa_onnx::OfflineTts> impl;
};

SherpaOnnxOfflineTts *SherpaOnnxCreateOfflineTts(
    const SherpaOnnxOfflineTtsConfig *config) {
  sherpa_onnx::OfflineTtsConfig tts_config;

  tts_config.model.vits.model = SHERPA_ONNX_OR(config->model.vits.model, "");
  tts_config.model.vits.lexicon =
      SHERPA_ONNX_OR(config->model.vits.lexicon, "");
  tts_config.model.vits.tokens = SHERPA_ONNX_OR(config->model.vits.tokens, "");
  tts_config.model.vits.data_dir =
      SHERPA_ONNX_OR(config->model.vits.data_dir, "");
  tts_config.model.vits.noise_scale =
      SHERPA_ONNX_OR(config->model.vits.noise_scale, 0.667);
  tts_config.model.vits.noise_scale_w =
      SHERPA_ONNX_OR(config->model.vits.noise_scale_w, 0.8);
  tts_config.model.vits.length_scale =
      SHERPA_ONNX_OR(config->model.vits.length_scale, 1.0);

  tts_config.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  tts_config.model.debug = config->model.debug;
  tts_config.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");
  tts_config.rule_fsts = SHERPA_ONNX_OR(config->rule_fsts, "");
  tts_config.max_num_sentences = SHERPA_ONNX_OR(config->max_num_sentences, 2);

  if (tts_config.model.debug) {
    fprintf(stderr, "%s\n", tts_config.ToString().c_str());
  }

  SherpaOnnxOfflineTts *tts = new SherpaOnnxOfflineTts;

  tts->impl = std::make_unique<sherpa_onnx::OfflineTts>(tts_config);

  return tts;
}

void SherpaOnnxDestroyOfflineTts(SherpaOnnxOfflineTts *tts) { delete tts; }

int32_t SherpaOnnxOfflineTtsSampleRate(const SherpaOnnxOfflineTts *tts) {
  return tts->impl->SampleRate();
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerate(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid,
    float speed) {
  return SherpaOnnxOfflineTtsGenerateWithCallback(tts, text, sid, speed,
                                                  nullptr);
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateWithCallback(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioCallback callback) {
  sherpa_onnx::GeneratedAudio audio =
      tts->impl->Generate(text, sid, speed, callback);

  if (audio.samples.empty()) {
    return nullptr;
  }

  SherpaOnnxGeneratedAudio *ans = new SherpaOnnxGeneratedAudio;

  float *samples = new float[audio.samples.size()];
  std::copy(audio.samples.begin(), audio.samples.end(), samples);

  ans->samples = samples;
  ans->n = audio.samples.size();
  ans->sample_rate = audio.sample_rate;

  return ans;
}

void SherpaOnnxDestroyOfflineTtsGeneratedAudio(
    const SherpaOnnxGeneratedAudio *p) {
  if (p) {
    delete[] p->samples;
    delete p;
  }
}

int32_t SherpaOnnxWriteWave(const float *samples, int32_t n,
                            int32_t sample_rate, const char *filename) {
  return sherpa_onnx::WriteWave(filename, sample_rate, samples, n);
}
