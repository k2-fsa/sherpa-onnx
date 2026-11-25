// c-api-examples/target-speaker-vad-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation

// This example shows how to build a target-speaker VAD with NOMO-PVAD.
//
// Unlike a regular VAD, which answers "is anybody speaking?", this model
// answers "is the *enrolled* speaker speaking?".
//
// You need two models:
//
// 1. A speaker embedding extractor, used once on an enrollment clip.
//    We use the 3d-speaker ERes2NetV2 model, which is the model nomo-pvad was
//    trained with.
//
//    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx
//
// 2. The target-speaker VAD model itself, exported with
//    scripts/nomo-pvad/export-onnx.py
//
// Usage:
//   ./target-speaker-vad-c-api \
//       ./3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx \
//       ./nomo-pvad.onnx \
//       ./enroll.wav \
//       ./test.wav

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

static const float *ComputeEmbedding(
    const SherpaOnnxSpeakerEmbeddingExtractor *ex, const char *wav_filename) {
  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    exit(-1);
  }

  const SherpaOnnxOnlineStream *stream =
      SherpaOnnxSpeakerEmbeddingExtractorCreateStream(ex);

  SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate, wave->samples,
                                       wave->num_samples);
  SherpaOnnxOnlineStreamInputFinished(stream);

  if (!SherpaOnnxSpeakerEmbeddingExtractorIsReady(ex, stream)) {
    fprintf(stderr, "The enrollment wave file %s is too short!\n",
            wav_filename);
    exit(-1);
  }

  const float *v =
      SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(ex, stream);

  SherpaOnnxDestroyOnlineStream(stream);
  SherpaOnnxFreeWave(wave);

  // Remember to free v with SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding()
  return v;
}

int32_t main(int32_t argc, char *argv[]) {
  const char *embedding_model =
      "./3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx";
  const char *pvad_model = "./nomo-pvad.onnx";
  const char *enroll_wav = "./enroll.wav";
  const char *test_wav = "./test.wav";

  if (argc == 5) {
    embedding_model = argv[1];
    pvad_model = argv[2];
    enroll_wav = argv[3];
    test_wav = argv[4];
  }

  // ---------------------------------------------------------------
  // Step 1: compute the enrollment embedding of the target speaker.
  // ---------------------------------------------------------------
  SherpaOnnxSpeakerEmbeddingExtractorConfig ex_config;
  memset(&ex_config, 0, sizeof(ex_config));
  ex_config.model = embedding_model;
  ex_config.num_threads = 1;
  ex_config.debug = 0;
  ex_config.provider = "cpu";

  const SherpaOnnxSpeakerEmbeddingExtractor *ex =
      SherpaOnnxCreateSpeakerEmbeddingExtractor(&ex_config);
  if (!ex) {
    fprintf(stderr, "Failed to create the speaker embedding extractor\n");
    return -1;
  }

  int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(ex);
  fprintf(stderr, "Speaker embedding dim: %d\n", dim);

  const float *target = ComputeEmbedding(ex, enroll_wav);
  fprintf(stderr, "Computed the enrollment embedding from %s\n", enroll_wav);

  // ---------------------------------------------------------------
  // Step 2: create the target-speaker VAD.
  // ---------------------------------------------------------------
  SherpaOnnxVadModelConfig config;
  memset(&config, 0, sizeof(config));

  config.nomo_pvad.model = pvad_model;
  config.nomo_pvad.speaker_embedding = target;
  config.nomo_pvad.speaker_embedding_dim = dim;
  config.nomo_pvad.threshold = 0.7f;
  config.nomo_pvad.min_silence_duration = 0.5f;
  config.nomo_pvad.min_speech_duration = 0.25f;
  config.nomo_pvad.max_speech_duration = 20.0f;

  config.sample_rate = 16000;
  config.num_threads = 2;
  config.provider = "cpu";
  config.debug = 1;

  const SherpaOnnxVoiceActivityDetector *vad =
      SherpaOnnxCreateVoiceActivityDetector(&config, 60);
  if (!vad) {
    fprintf(stderr, "Failed to create the target-speaker VAD\n");
    return -1;
  }

  SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(target);
  SherpaOnnxDestroySpeakerEmbeddingExtractor(ex);

  // ---------------------------------------------------------------
  // Step 3: feed the audio stream and collect speech segments.
  // ---------------------------------------------------------------
  const SherpaOnnxWave *wave = SherpaOnnxReadWave(test_wav);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", test_wav);
    return -1;
  }

  int32_t sample_rate = wave->sample_rate;
  int32_t num_samples = wave->num_samples;

  // NOMO-PVAD supports only 16 kHz, which is also what config.sample_rate is
  // set to above. Resample your audio first if it uses another rate.
  if (sample_rate != config.sample_rate) {
    fprintf(stderr, "Expected %d Hz but %s is %d Hz. Please resample it.\n",
            config.sample_rate, test_wav, sample_rate);
    SherpaOnnxFreeWave(wave);
    SherpaOnnxDestroyVoiceActivityDetector(vad);
    return -1;
  }

  fprintf(stderr, "Processing %s: %.2f seconds at %d Hz\n", test_wav,
          (float)num_samples / sample_rate, sample_rate);

  // Feed 20 ms at a time, like a microphone would.
  const int32_t frame_size = sample_rate / 50;

  int32_t num_segments = 0;
  float total_speech = 0;

  for (int32_t start = 0; start < num_samples; start += frame_size) {
    int32_t n = frame_size;
    if (start + n > num_samples) {
      n = num_samples - start;
    }

    SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, wave->samples + start,
                                                  n);

    while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
      const SherpaOnnxSpeechSegment *segment =
          SherpaOnnxVoiceActivityDetectorFront(vad);

      float t0 = (float)segment->start / sample_rate;
      float t1 = (float)(segment->start + segment->n) / sample_rate;

      printf("segment %d: [%.2f, %.2f] s (%.2f s)\n", num_segments, t0, t1,
             t1 - t0);

      total_speech += t1 - t0;
      ++num_segments;

      SherpaOnnxDestroySpeechSegment(segment);
      SherpaOnnxVoiceActivityDetectorPop(vad);
    }
  }

  SherpaOnnxVoiceActivityDetectorFlush(vad);

  while (!SherpaOnnxVoiceActivityDetectorEmpty(vad)) {
    const SherpaOnnxSpeechSegment *segment =
        SherpaOnnxVoiceActivityDetectorFront(vad);

    float t0 = (float)segment->start / sample_rate;
    float t1 = (float)(segment->start + segment->n) / sample_rate;

    printf("segment %d: [%.2f, %.2f] s (%.2f s)\n", num_segments, t0, t1,
           t1 - t0);

    total_speech += t1 - t0;
    ++num_segments;

    SherpaOnnxDestroySpeechSegment(segment);
    SherpaOnnxVoiceActivityDetectorPop(vad);
  }

  printf("\nDetected %d segment(s), %.2f s of target speech out of %.2f s\n",
         num_segments, total_speech, (float)num_samples / sample_rate);

  SherpaOnnxFreeWave(wave);
  SherpaOnnxDestroyVoiceActivityDetector(vad);

  return 0;
}
