/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
  if (argc < 8 || argc > 9) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-onnx \
    /path/to/tokens.txt \
    /path/to/encoder.onnx \
    /path/to/decoder.onnx \
    /path/to/joiner.onnx \
    /path/to/joiner_encoder_proj.onnx \
    /path/to/joiner_decoder_proj.onnx \
    /path/to/foo.wav [num_threads]

You can download pre-trained models from the following repository:
https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
)usage";
    std::cerr << usage << "\n";

    return 0;
  }

  std::string tokens = argv[1];
  std::string encoder = argv[2];
  std::string decoder = argv[3];
  std::string joiner = argv[4];
  std::string joiner_encoder_proj = argv[5];
  std::string joiner_decoder_proj = argv[6];
  std::string wav_filename = argv[7];
  int32_t num_threads = 4;
  if (argc == 9) {
    num_threads = atoi(argv[8]);
  }

  sherpa_onnx::SymbolTable sym(tokens);

  int32_t num_frames;
  auto features = ComputeFeatures(wav_filename, 16000, &num_frames);
  int32_t feature_dim = features.size() / num_frames;

  sherpa_onnx::RnntModel model(encoder, decoder, joiner, joiner_encoder_proj,
                               joiner_decoder_proj, num_threads);
  Ort::Value encoder_out =
      model.RunEncoder(features.data(), num_frames, feature_dim);

  auto hyp = sherpa_onnx::GreedySearch(model, encoder_out);

  std::string text;
  for (auto i : hyp) {
    text += sym[i];
  }

  std::cout << "Recognition result for " << wav_filename << "\n"
            << text << "\n";

  return 0;
}
