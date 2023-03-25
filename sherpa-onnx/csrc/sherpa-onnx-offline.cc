// sherpa-onnx/csrc/sherpa-onnx-offline.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>

#include <memory>

#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-model.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  sherpa_onnx::OfflineTransducerModelConfig model_config;
  model_config.encoder_filename =
      "./sherpa-onnx-conformer-en-2023-03-18/encoder-epoch-99-avg-1.onnx";
  model_config.decoder_filename =
      "./sherpa-onnx-conformer-en-2023-03-18/decoder-epoch-99-avg-1.onnx";
  model_config.joiner_filename =
      "./sherpa-onnx-conformer-en-2023-03-18/joiner-epoch-99-avg-1.onnx";
  model_config.tokens = "./sherpa-onnx-conformer-en-2023-03-18/tokens.txt";
  model_config.debug = false;
  std::string wav_filename0 =
      "./sherpa-onnx-conformer-en-2023-03-18/test_wavs/0.wav";
  std::string wav_filename1 =
      "./sherpa-onnx-conformer-en-2023-03-18/test_wavs/1.wav";
  std::string wav_filename2 =
      "./sherpa-onnx-conformer-en-2023-03-18/test_wavs/2.wav";

  int32_t expected_sampling_rate = 16000;

  bool is_ok = false;

  // TODO(fangjun): Change ReadWave() to also return sampling rate
  std::vector<float> samples0 =
      sherpa_onnx::ReadWave(wav_filename0, expected_sampling_rate, &is_ok);
  assert(is_ok);

  std::vector<float> samples1 =
      sherpa_onnx::ReadWave(wav_filename1, expected_sampling_rate, &is_ok);
  assert(is_ok);

  std::vector<float> samples2 =
      sherpa_onnx::ReadWave(wav_filename2, expected_sampling_rate, &is_ok);
  assert(is_ok);

  sherpa_onnx::OfflineRecognizerConfig config;
  config.model_config = model_config;

  sherpa_onnx::OfflineRecognizer recognizer(config);
  auto s0 = recognizer.CreateStream();
  auto s1 = recognizer.CreateStream();
  auto s2 = recognizer.CreateStream();

  s0->AcceptWaveform(expected_sampling_rate, samples0.data(), samples0.size());
  s1->AcceptWaveform(expected_sampling_rate, samples1.data(), samples1.size());
  s2->AcceptWaveform(expected_sampling_rate, samples2.data(), samples2.size());

  std::vector<sherpa_onnx::OfflineStream *> ss = {s1.get(), s2.get()};

  // decode a single stream
  recognizer.DecodeStream(s0.get());

  // decode multiple streams in parallel
  recognizer.DecodeStreams(ss.data(), ss.size());

  fprintf(stderr, "%s\n%s\n\n", wav_filename0.c_str(),
          s0->GetResult().text.c_str());

  fprintf(stderr, "%s\n%s\n\n", wav_filename1.c_str(),
          s1->GetResult().text.c_str());

  fprintf(stderr, "%s\n%s\n\n", wav_filename2.c_str(),
          s2->GetResult().text.c_str());

  return 0;
}
