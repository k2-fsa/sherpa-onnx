// c-api-examples/forced-alignment-en-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates forced alignment of English speech with
// sherpa-onnx's C API. It aligns the test wav of the english-mfa-v3.1.0
// model package with its transcript and writes a Praat TextGrid.
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
tar xvf english-mfa-v3.1.0.tar.bz2
rm english-mfa-v3.1.0.tar.bz2
*/
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  const char* model_dir = "./english-mfa-v3.1.0";
  const char* wav_filename = "./english-mfa-v3.1.0/en.wav";

  const char* text =
      "ask not what your country can do for you, "
      "ask what you can do for your country.";

  const SherpaOnnxWave* wave = SherpaOnnxReadWave(wav_filename);
  if (!wave) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  char lexicon[256];
  snprintf(lexicon, sizeof(lexicon), "%s/lexicon.txt", model_dir);

  SherpaOnnxForcedAlignerConfig config;
  memset(&config, 0, sizeof(config));
  config.model_dir = model_dir;
  config.lexicon = lexicon;

  SherpaOnnxForcedAligner* aligner = SherpaOnnxCreateForcedAligner(&config);
  if (!aligner) {
    fprintf(stderr, "Failed to create the forced aligner!\n");
    SherpaOnnxFreeWave(wave);
    return -1;
  }

  SherpaOnnxOfflineStream* stream =
      SherpaOnnxForcedAlignerCreateStream(aligner);

  // Optional: per-stream options, e.g. debug output of one utterance.
  SherpaOnnxOfflineStreamSetOption(stream, "debug", "1");

  SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                  wave->num_samples);

  const SherpaOnnxForcedAlignmentResult* result =
      SherpaOnnxForcedAlignerCompute(aligner, stream, text);

  if (result) {
    printf("duration: %.3f seconds\n", result->duration);

    for (int32_t i = 0; i != result->num_tiers; ++i) {
      const SherpaOnnxForcedAlignmentTier* t = &result->tiers[i];
      printf("tier %s: %d intervals\n", t->name, t->num_intervals);
    }

    const char* textgrid_filename = "en.TextGrid";
    FILE* fp = fopen(textgrid_filename, "wb");
    fprintf(fp, "%s", result->textgrid);
    fclose(fp);
    printf("\n%s created!\n", textgrid_filename);

    SherpaOnnxDestroyForcedAlignmentResult(result);
  }

  SherpaOnnxDestroyOfflineStream(stream);
  SherpaOnnxDestroyForcedAligner(aligner);
  SherpaOnnxFreeWave(wave);

  return 0;
}
