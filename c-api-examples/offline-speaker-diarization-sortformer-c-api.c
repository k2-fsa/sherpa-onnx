// c-api-examples/offline-speaker-diarization-sortformer-c-api.c
//
// Copyright (c)  2026  Silvio Tomatis

//
// This file demonstrates how to implement speaker diarization with
// Nemotron-3-Diarization, an end-to-end streaming Sortformer model.
// It needs neither a speaker embedding model nor clustering.

/*
Usage:

Step 1: Download the model

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-nemotron-3-diarization.tar.bz2
  tar xvf sherpa-onnx-nemotron-3-diarization.tar.bz2
  rm sherpa-onnx-nemotron-3-diarization.tar.bz2

Step 2. Download test wave files

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Step 3. Run it

 */

#include <stdio.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int main() {
  const char *model = "./sherpa-onnx-nemotron-3-diarization/model.int8.onnx";
  const char *wav_filename = "./0-four-speakers-zh.wav";

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  SherpaOnnxOfflineSpeakerDiarizationConfig config;
  memset(&config, 0, sizeof(config));

  config.segmentation.sortformer.model = model;
  // A value of 0 also uses the default of 0.5.
  config.segmentation.sortformer.threshold = 0.5f;
  config.segmentation.num_threads = 2;

  const SherpaOnnxOfflineSpeakerDiarization *sd =
      SherpaOnnxCreateOfflineSpeakerDiarization(&config);

  if (!sd) {
    fprintf(stderr, "Failed to initialize offline speaker diarization\n");
    SherpaOnnxFreeWave(wave);
    return -1;
  }

  const SherpaOnnxOfflineSpeakerDiarizationResult *result = NULL;
  const SherpaOnnxOfflineSpeakerDiarizationSegment *segments = NULL;

  if (SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(sd) !=
      wave->sample_rate) {
    fprintf(
        stderr,
        "Expected sample rate: %d. Actual sample rate from the wave file: %d\n",
        SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(sd),
        wave->sample_rate);
    goto failed;
  }

  result = SherpaOnnxOfflineSpeakerDiarizationProcess(sd, wave->samples,
                                                       wave->num_samples);
  if (!result) {
    fprintf(stderr, "Failed to do speaker diarization");
    goto failed;
  }

  int32_t num_speakers =
      SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(result);
  int32_t num_segments =
      SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);
  fprintf(stderr, "Number of speakers: %d\n", num_speakers);

  segments = SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);

  for (int32_t i = 0; i != num_segments; ++i) {
    fprintf(stderr, "%.3f -- %.3f speaker_%02d\n", segments[i].start,
            segments[i].end, segments[i].speaker);
  }

failed:

  SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments);
  SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);
  SherpaOnnxDestroyOfflineSpeakerDiarization(sd);
  SherpaOnnxFreeWave(wave);

  return 0;
}
