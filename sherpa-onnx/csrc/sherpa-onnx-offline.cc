// sherpa-onnx/csrc/sherpa-onnx-offline.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>

#include <iostream>
#include <memory>

#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-model.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  sherpa_onnx::OfflineTransducerModelConfig config;
  config.encoder_filename =
      "./sherpa-onnx-conformer-en-2023-03-18/encoder-epoch-99-avg-1.onnx";
  config.decoder_filename =
      "./sherpa-onnx-conformer-en-2023-03-18/decoder-epoch-99-avg-1.onnx";
  config.joiner_filename =
      "./sherpa-onnx-conformer-en-2023-03-18/joiner-epoch-99-avg-1.onnx";
  config.tokens = "./sherpa-onnx-conformer-en-2023-03-18/tokens.txt";
  config.debug = true;
  std::string wav_filename0 =
      "./sherpa-onnx-conformer-en-2023-03-18/test_wavs/0.wav";
  std::string wav_filename1 =
      "./sherpa-onnx-conformer-en-2023-03-18/test_wavs/1.wav";
  std::string wav_filename2 =
      "./sherpa-onnx-conformer-en-2023-03-18/test_wavs/2.wav";

  sherpa_onnx::OfflineTransducerModel model(config);

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

  sherpa_onnx::OfflineStream stream0;
  stream0.AcceptWaveform(expected_sampling_rate, samples0.data(),
                         samples0.size());

  sherpa_onnx::OfflineStream stream1;
  stream1.AcceptWaveform(expected_sampling_rate, samples1.data(),
                         samples1.size());

  sherpa_onnx::OfflineStream stream2;
  stream2.AcceptWaveform(expected_sampling_rate, samples2.data(),
                         samples2.size());

  auto features0 = stream0.GetFrames();
  auto features1 = stream1.GetFrames();
  auto features2 = stream2.GetFrames();

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<int64_t, 2> shape0 = {
      static_cast<int32_t>(features0.size()) / stream0.FeatureDim(),  // T
      stream0.FeatureDim()                                            // C
  };

  assert(shape0[0] * shape0[1] == features0.size());
  Ort::Value x0 =
      Ort::Value::CreateTensor(memory_info, features0.data(), features0.size(),
                               shape0.data(), shape0.size());

  std::array<int64_t, 2> shape1 = {
      static_cast<int32_t>(features1.size()) / stream1.FeatureDim(),  // T
      stream1.FeatureDim()                                            // C
  };

  assert(shape1[0] * shape1[1] == features1.size());
  Ort::Value x1 =
      Ort::Value::CreateTensor(memory_info, features1.data(), features1.size(),
                               shape1.data(), shape1.size());

  std::array<int64_t, 2> shape2 = {
      static_cast<int32_t>(features2.size()) / stream2.FeatureDim(),  // T
      stream2.FeatureDim()                                            // C
  };

  assert(shape2[0] * shape2[1] == features2.size());
  Ort::Value x2 =
      Ort::Value::CreateTensor(memory_info, features2.data(), features2.size(),
                               shape2.data(), shape2.size());

  std::array<int64_t, 3> features_length = {shape0[0], shape1[0], shape2[0]};

  std::array<int64_t, 1> features_shape = {3};

  Ort::Value x = sherpa_onnx::PadSequence(model.Allocator(), {&x0, &x1, &x2},
                                          -23.025850929940457f);

  Ort::Value x_length = Ort::Value::CreateTensor(
      memory_info, features_length.data(), features_length.size(),
      features_shape.data(), features_shape.size());

  auto t = model.RunEncoder(std::move(x), std::move(x_length));

  std::unique_ptr<sherpa_onnx::OfflineTransducerDecoder> decoder =
      std::make_unique<sherpa_onnx::OfflineTransducerGreedySearchDecoder>(
          &model);
  auto results = decoder->Decode(std::move(t.first), std::move(t.second));

  sherpa_onnx::SymbolTable sym(config.tokens);
  std::vector<std::string> text_vec;
  for (const auto &r : results) {
    std::string text;
    for (const auto &t : r.tokens) {
      text.append(sym[t]);
    }
    text_vec.push_back(std::move(text));
  }

  std::cout << wav_filename0 << "\n" << text_vec[0] << "\n\n";
  std::cout << wav_filename1 << "\n" << text_vec[1] << "\n\n";
  std::cout << wav_filename2 << "\n" << text_vec[2] << "\n\n";

  return 0;
}
