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
  std::string wav_filename =
      "./sherpa-onnx-conformer-en-2023-03-18/test_wavs/0.wav";

  sherpa_onnx::OfflineTransducerModel model(config);

  int32_t expected_sampling_rate = 16000;

  bool is_ok = false;

  // TODO(fangjun): Change ReadWave() to also return sampling rate
  std::vector<float> samples =
      sherpa_onnx::ReadWave(wav_filename, expected_sampling_rate, &is_ok);
  std::cout << "samples size: " << samples.size() << "\n";

  sherpa_onnx::OfflineStream stream;
  stream.AcceptWaveform(expected_sampling_rate, samples.data(), samples.size());
  auto features = stream.GetFrames();
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<int64_t, 3> shape = {
      1,                                                            // batch
      static_cast<int32_t>(features.size()) / stream.FeatureDim(),  // T
      stream.FeatureDim()                                           // C
  };

  assert(shape[1] * shape[2] == features.size());
  Ort::Value x =
      Ort::Value::CreateTensor(memory_info, features.data(), features.size(),
                               shape.data(), shape.size());

  std::array<int64_t, 1> features_length = {shape[1]};
  std::array<int64_t, 1> features_shape = {1};

  Ort::Value x_length = Ort::Value::CreateTensor(
      memory_info, features_length.data(), features_length.size(),
      features_shape.data(), features_shape.size());
  auto t = model.RunEncoder(std::move(x), std::move(x_length));

  for (auto i : t.first.GetTensorTypeAndShapeInfo().GetShape()) {
    std::cout << i << " ";
  }
  std::cout << "\n";

  for (auto i : t.second.GetTensorTypeAndShapeInfo().GetShape()) {
    std::cout << i << " ";
  }
  std::cout << "\n";

  std::cout << t.second.GetTensorData<int64_t>()[0] << "\n";
  std::unique_ptr<sherpa_onnx::OfflineTransducerDecoder> decoder =
      std::make_unique<sherpa_onnx::OfflineTransducerGreedySearchDecoder>(
          &model);
  auto results = decoder->Decode(std::move(t.first), std::move(t.second));

  sherpa_onnx::SymbolTable sym(config.tokens);
  std::string text;
  for (const auto &t : results[0].tokens) {
    text.append(sym[t]);
  }
  std::cout << text << "\n";

  return 0;
}
