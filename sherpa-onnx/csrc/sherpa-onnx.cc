// sherpa-onnx/csrc/sherpa-onnx.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <iostream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "sherpa-onnx/csrc/decode.h"
#include "sherpa-onnx/csrc/rnnt-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/wave-reader.h"

/** Compute fbank features of the input wave filename.
 *
 * @param wav_filename. Path to a mono wave file.
 * @param expected_sampling_rate  Expected sampling rate of the input wave file.
 * @param num_frames On return, it contains the number of feature frames.
 * @return Return the computed feature of shape (num_frames, feature_dim)
 *         stored in row-major.
 */
static std::vector<float> ComputeFeatures(const std::string &wav_filename,
                                          float expected_sampling_rate,
                                          int32_t *num_frames) {
  std::vector<float> samples =
      sherpa_onnx::ReadWave(wav_filename, expected_sampling_rate);

  float duration = samples.size() / expected_sampling_rate;

  std::cout << "wav filename: " << wav_filename << "\n";
  std::cout << "wav duration (s): " << duration << "\n";

  knf::FbankOptions opts;
  opts.frame_opts.dither = 0;
  opts.frame_opts.snip_edges = false;
  opts.frame_opts.samp_freq = expected_sampling_rate;

  int32_t feature_dim = 80;

  opts.mel_opts.num_bins = feature_dim;

  knf::OnlineFbank fbank(opts);
  fbank.AcceptWaveform(expected_sampling_rate, samples.data(), samples.size());
  fbank.InputFinished();

  *num_frames = fbank.NumFramesReady();

  std::vector<float> features(*num_frames * feature_dim);
  float *p = features.data();

  for (int32_t i = 0; i != fbank.NumFramesReady(); ++i, p += feature_dim) {
    const float *f = fbank.GetFrame(i);
    std::copy(f, f + feature_dim, p);
  }

  return features;
}

int main(int32_t argc, char *argv[]) {
  if (argc < 6 || argc > 7) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-onnx \
    /path/to/tokens.txt \
    /path/to/encoder.onnx \
    /path/to/decoder.onnx \
    /path/to/joiner.onnx \
    /path/to/foo.wav [num_threads]

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";
    std::cerr << usage << "\n";

    return 0;
  }

  std::string tokens = argv[1];
  std::string encoder = argv[2];
  std::string decoder = argv[3];
  std::string joiner = argv[4];
  std::string wav_filename = argv[5];
  int32_t num_threads = 4;
  if (argc == 6) {
    num_threads = atoi(argv[8]);
  }
  fprintf(stderr, "num_threads: %d\n", num_threads);

  sherpa_onnx::SymbolTable sym(tokens);

  int32_t num_frames;
  auto features = ComputeFeatures(wav_filename, 16000, &num_frames);
  int32_t feature_dim = features.size() / num_frames;

  sherpa_onnx::RnntModel model(encoder, decoder, joiner, num_threads);
  fprintf(stderr, "here0\n");
  Ort::Value encoder_out =
      model.RunEncoder(features.data(), num_frames, feature_dim);
  fprintf(stderr, "here00\n");

  auto hyp = sherpa_onnx::GreedySearch(model, encoder_out);

  std::string text;
  for (auto i : hyp) {
    text += sym[i];
  }

  std::cout << "Recognition result for " << wav_filename << "\n"
            << text << "\n";

  return 0;
}
