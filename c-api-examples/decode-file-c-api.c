// c-api-examples/decode-file-c-api.c
//
// Copyright (c)  2023  Xiaomi Corporation

// This file shows how to use sherpa-onnx C API
// to decode a file.
//
// clang-format off
/*
Usage

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
tar xf sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
rm sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2

./decode-file-c-api

 */
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  const char* wav_filename =
      "./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/"
      "test_wavs/0.wav";
  const char* tokens =
      "./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/"
      "tokens.txt";
  const char* encoder =
      "./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/"
      "encoder.onnx";
  const char* decoder =
      "./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/"
      "decoder.onnx";
  const char* joiner =
      "./sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms/"
      "joiner.onnx";

  SherpaOnnxOnlineRecognizerConfig config;
  memset(&config, 0, sizeof(config));

  config.model_config.debug = 0;
  config.model_config.num_threads = 2;
  config.model_config.provider = "cpu";
  config.model_config.tokens = tokens;
  config.model_config.transducer.encoder = encoder;
  config.model_config.transducer.decoder = decoder;
  config.model_config.transducer.joiner = joiner;

  config.decoding_method = "greedy_search";

  config.max_active_paths = 4;

  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;

  config.enable_endpoint = 1;
  config.rule1_min_trailing_silence = 2.4;
  config.rule2_min_trailing_silence = 1.2;
  config.rule3_min_utterance_length = 300;

  const SherpaOnnxOnlineRecognizer* recognizer =
      SherpaOnnxCreateOnlineRecognizer(&config);
  const SherpaOnnxOnlineStream* stream =
      SherpaOnnxCreateOnlineStream(recognizer);

  const SherpaOnnxDisplay* display = SherpaOnnxCreateDisplay(50);
  int32_t segment_id = 0;

  const SherpaOnnxWave* wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

#define N 3200  // 0.2 s. Sample rate is fixed to 16 kHz

  fprintf(stderr, "sample rate: %d, num samples: %d, duration: %.2f s\n",
          wave->sample_rate, wave->num_samples,
          (float)wave->num_samples / wave->sample_rate);

  int32_t k = 0;
  while (k < wave->num_samples) {
    int32_t start = k;
    int32_t end =
        (start + N > wave->num_samples) ? wave->num_samples : (start + N);
    k += N;

    SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate,
                                         wave->samples + start, end - start);
    while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
      SherpaOnnxDecodeOnlineStream(recognizer, stream);
    }

    const SherpaOnnxOnlineRecognizerResult* r =
        SherpaOnnxGetOnlineStreamResult(recognizer, stream);

    if (strlen(r->text)) {
      SherpaOnnxPrint(display, segment_id, r->text);
    }

    if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream)) {
      if (strlen(r->text)) {
        ++segment_id;
      }
      SherpaOnnxOnlineStreamReset(recognizer, stream);
    }

    SherpaOnnxDestroyOnlineRecognizerResult(r);
  }

  // add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail_paddings,
                                       4800);

  SherpaOnnxFreeWave(wave);

  SherpaOnnxOnlineStreamInputFinished(stream);
  while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
    SherpaOnnxDecodeOnlineStream(recognizer, stream);
  }

  const SherpaOnnxOnlineRecognizerResult* r =
      SherpaOnnxGetOnlineStreamResult(recognizer, stream);

  if (strlen(r->text)) {
    SherpaOnnxPrint(display, segment_id, r->text);
  }

  SherpaOnnxDestroyOnlineRecognizerResult(r);

  SherpaOnnxDestroyDisplay(display);
  SherpaOnnxDestroyOnlineStream(stream);
  SherpaOnnxDestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
