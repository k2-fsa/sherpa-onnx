// sherpa-onnx/c-api/cxx-api.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-onnx/c-api/cxx-api.h"

#include <algorithm>
#include <cstring>

namespace sherpa_onnx::cxx {

Wave ReadWave(const std::string &filename) {
  auto p = SherpaOnnxReadWave(filename.c_str());

  Wave ans;
  if (p) {
    ans.samples.resize(p->num_samples);

    std::copy(p->samples, p->samples + p->num_samples, ans.samples.data());

    ans.sample_rate = p->sample_rate;
    SherpaOnnxFreeWave(p);
  }

  return ans;
}

OnlineStream::OnlineStream(const SherpaOnnxOnlineStream *p)
    : MoveOnly<OnlineStream, SherpaOnnxOnlineStream>(p) {}

void OnlineStream::Destroy(const SherpaOnnxOnlineStream *p) const {
  SherpaOnnxDestroyOnlineStream(p);
}

void OnlineStream::AcceptWaveform(int32_t sample_rate, const float *samples,
                                  int32_t n) const {
  SherpaOnnxOnlineStreamAcceptWaveform(p_, sample_rate, samples, n);
}

void OnlineStream::InputFinished() const {
  SherpaOnnxOnlineStreamInputFinished(p_);
}

OnlineRecognizer OnlineRecognizer::Create(
    const OnlineRecognizerConfig &config) {
  struct SherpaOnnxOnlineRecognizerConfig c;
  memset(&c, 0, sizeof(c));

  c.feat_config.sample_rate = config.feat_config.sample_rate;
  c.feat_config.feature_dim = config.feat_config.feature_dim;

  c.model_config.transducer.encoder =
      config.model_config.transducer.encoder.c_str();
  c.model_config.transducer.decoder =
      config.model_config.transducer.decoder.c_str();
  c.model_config.transducer.joiner =
      config.model_config.transducer.joiner.c_str();

  c.model_config.paraformer.encoder =
      config.model_config.paraformer.encoder.c_str();
  c.model_config.paraformer.decoder =
      config.model_config.paraformer.decoder.c_str();

  c.model_config.zipformer2_ctc.model =
      config.model_config.zipformer2_ctc.model.c_str();

  c.model_config.tokens = config.model_config.tokens.c_str();
  c.model_config.num_threads = config.model_config.num_threads;
  c.model_config.provider = config.model_config.provider.c_str();
  c.model_config.debug = config.model_config.debug;
  c.model_config.model_type = config.model_config.model_type.c_str();
  c.model_config.modeling_unit = config.model_config.modeling_unit.c_str();
  c.model_config.bpe_vocab = config.model_config.bpe_vocab.c_str();
  c.model_config.tokens_buf = config.model_config.tokens_buf.c_str();
  c.model_config.tokens_buf_size = config.model_config.tokens_buf.size();

  c.decoding_method = config.decoding_method.c_str();
  c.max_active_paths = config.max_active_paths;
  c.enable_endpoint = config.enable_endpoint;
  c.rule1_min_trailing_silence = config.rule1_min_trailing_silence;
  c.rule2_min_trailing_silence = config.rule2_min_trailing_silence;
  c.rule3_min_utterance_length = config.rule3_min_utterance_length;
  c.hotwords_file = config.hotwords_file.c_str();
  c.hotwords_score = config.hotwords_score;

  c.ctc_fst_decoder_config.graph = config.ctc_fst_decoder_config.graph.c_str();
  c.ctc_fst_decoder_config.max_active =
      config.ctc_fst_decoder_config.max_active;

  c.rule_fsts = config.rule_fsts.c_str();
  c.rule_fars = config.rule_fars.c_str();

  c.blank_penalty = config.blank_penalty;

  c.hotwords_buf = config.hotwords_buf.c_str();
  c.hotwords_buf_size = config.hotwords_buf.size();

  auto p = SherpaOnnxCreateOnlineRecognizer(&c);
  return OnlineRecognizer(p);
}

OnlineRecognizer::OnlineRecognizer(const SherpaOnnxOnlineRecognizer *p)
    : MoveOnly<OnlineRecognizer, SherpaOnnxOnlineRecognizer>(p) {}

void OnlineRecognizer::Destroy(const SherpaOnnxOnlineRecognizer *p) const {
  SherpaOnnxDestroyOnlineRecognizer(p);
}

OnlineStream OnlineRecognizer::CreateStream() const {
  auto s = SherpaOnnxCreateOnlineStream(p_);
  return OnlineStream{s};
}

OnlineStream OnlineRecognizer::CreateStream(const std::string &hotwords) const {
  auto s = SherpaOnnxCreateOnlineStreamWithHotwords(p_, hotwords.c_str());
  return OnlineStream{s};
}

bool OnlineRecognizer::IsReady(const OnlineStream *s) const {
  return SherpaOnnxIsOnlineStreamReady(p_, s->Get());
}

void OnlineRecognizer::Decode(const OnlineStream *s) const {
  SherpaOnnxDecodeOnlineStream(p_, s->Get());
}

void OnlineRecognizer::Reset(const OnlineStream *s) const {
  SherpaOnnxOnlineStreamReset(p_, s->Get());
}

bool OnlineRecognizer::IsEndpoint(const OnlineStream *s) const {
  return SherpaOnnxOnlineStreamIsEndpoint(p_, s->Get());
}

void OnlineRecognizer::Decode(const OnlineStream *ss, int32_t n) const {
  if (n <= 0) {
    return;
  }

  std::vector<const SherpaOnnxOnlineStream *> streams(n);
  for (int32_t i = 0; i != n; ++n) {
    streams[i] = ss[i].Get();
  }

  SherpaOnnxDecodeMultipleOnlineStreams(p_, streams.data(), n);
}

OnlineRecognizerResult OnlineRecognizer::GetResult(
    const OnlineStream *s) const {
  auto r = SherpaOnnxGetOnlineStreamResult(p_, s->Get());

  OnlineRecognizerResult ans;
  ans.text = r->text;

  ans.tokens.resize(r->count);
  for (int32_t i = 0; i != r->count; ++i) {
    ans.tokens[i] = r->tokens_arr[i];
  }

  if (r->timestamps) {
    ans.timestamps.resize(r->count);
    std::copy(r->timestamps, r->timestamps + r->count, ans.timestamps.data());
  }

  ans.json = r->json;

  SherpaOnnxDestroyOnlineRecognizerResult(r);

  return ans;
}

// ============================================================================
// Non-streaming ASR
// ============================================================================
OfflineStream::OfflineStream(const SherpaOnnxOfflineStream *p)
    : MoveOnly<OfflineStream, SherpaOnnxOfflineStream>(p) {}

void OfflineStream::Destroy(const SherpaOnnxOfflineStream *p) const {
  SherpaOnnxDestroyOfflineStream(p);
}

void OfflineStream::AcceptWaveform(int32_t sample_rate, const float *samples,
                                   int32_t n) const {
  SherpaOnnxAcceptWaveformOffline(p_, sample_rate, samples, n);
}

OfflineRecognizer OfflineRecognizer::Create(
    const OfflineRecognizerConfig &config) {
  struct SherpaOnnxOfflineRecognizerConfig c;
  memset(&c, 0, sizeof(c));

  c.feat_config.sample_rate = config.feat_config.sample_rate;
  c.feat_config.feature_dim = config.feat_config.feature_dim;
  c.model_config.transducer.encoder =
      config.model_config.transducer.encoder.c_str();
  c.model_config.transducer.decoder =
      config.model_config.transducer.decoder.c_str();
  c.model_config.transducer.joiner =
      config.model_config.transducer.joiner.c_str();

  c.model_config.paraformer.model =
      config.model_config.paraformer.model.c_str();

  c.model_config.nemo_ctc.model = config.model_config.nemo_ctc.model.c_str();

  c.model_config.whisper.encoder = config.model_config.whisper.encoder.c_str();
  c.model_config.whisper.decoder = config.model_config.whisper.decoder.c_str();
  c.model_config.whisper.language =
      config.model_config.whisper.language.c_str();
  c.model_config.whisper.task = config.model_config.whisper.task.c_str();
  c.model_config.whisper.tail_paddings =
      config.model_config.whisper.tail_paddings;

  c.model_config.tdnn.model = config.model_config.tdnn.model.c_str();

  c.model_config.tokens = config.model_config.tokens.c_str();
  c.model_config.num_threads = config.model_config.num_threads;
  c.model_config.debug = config.model_config.debug;
  c.model_config.provider = config.model_config.provider.c_str();
  c.model_config.model_type = config.model_config.model_type.c_str();
  c.model_config.modeling_unit = config.model_config.modeling_unit.c_str();
  c.model_config.bpe_vocab = config.model_config.bpe_vocab.c_str();
  c.model_config.telespeech_ctc = config.model_config.telespeech_ctc.c_str();

  c.model_config.sense_voice.model =
      config.model_config.sense_voice.model.c_str();
  c.model_config.sense_voice.language =
      config.model_config.sense_voice.language.c_str();
  c.model_config.sense_voice.use_itn = config.model_config.sense_voice.use_itn;

  c.lm_config.model = config.lm_config.model.c_str();
  c.lm_config.scale = config.lm_config.scale;

  c.decoding_method = config.decoding_method.c_str();
  c.max_active_paths = config.max_active_paths;
  c.hotwords_file = config.hotwords_file.c_str();
  c.hotwords_score = config.hotwords_score;

  c.rule_fsts = config.rule_fsts.c_str();
  c.rule_fars = config.rule_fars.c_str();

  c.blank_penalty = config.blank_penalty;

  auto p = SherpaOnnxCreateOfflineRecognizer(&c);
  return OfflineRecognizer(p);
}

OfflineRecognizer::OfflineRecognizer(const SherpaOnnxOfflineRecognizer *p)
    : MoveOnly<OfflineRecognizer, SherpaOnnxOfflineRecognizer>(p) {}

void OfflineRecognizer::Destroy(const SherpaOnnxOfflineRecognizer *p) const {
  SherpaOnnxDestroyOfflineRecognizer(p_);
}

OfflineStream OfflineRecognizer::CreateStream() const {
  auto p = SherpaOnnxCreateOfflineStream(p_);
  return OfflineStream{p};
}

void OfflineRecognizer::Decode(const OfflineStream *s) const {
  SherpaOnnxDecodeOfflineStream(p_, s->Get());
}

void OfflineRecognizer::Decode(const OfflineStream *ss, int32_t n) const {
  if (n <= 0) {
    return;
  }

  std::vector<const SherpaOnnxOfflineStream *> streams(n);
  for (int32_t i = 0; i != n; ++i) {
    streams[i] = ss[i].Get();
  }

  SherpaOnnxDecodeMultipleOfflineStreams(p_, streams.data(), n);
}

OfflineRecognizerResult OfflineRecognizer::GetResult(
    const OfflineStream *s) const {
  auto r = SherpaOnnxGetOfflineStreamResult(s->Get());

  OfflineRecognizerResult ans;
  if (r) {
    ans.text = r->text;

    if (r->timestamps) {
      ans.timestamps.resize(r->count);
      std::copy(r->timestamps, r->timestamps + r->count, ans.timestamps.data());
    }

    ans.tokens.resize(r->count);
    for (int32_t i = 0; i != r->count; ++i) {
      ans.tokens[i] = r->tokens_arr[i];
    }

    ans.json = r->json;
    ans.lang = r->lang ? r->lang : "";
    ans.emotion = r->emotion ? r->emotion : "";
    ans.event = r->event ? r->event : "";
  }

  SherpaOnnxDestroyOfflineRecognizerResult(r);

  return ans;
}

}  // namespace sherpa_onnx::cxx
