// sherpa-onnx/csrc/sherpa-onnx-offline-speaker-diarization.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speaker-diarization.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Offline/Non-streaming speaker diarization with sherpa-onnx
Usage example:

  )usage";
  sherpa_onnx::OfflineSpeakerDiarizationConfig config;
  sherpa_onnx::ParseOptions po(kUsageMessage);
  config.Register(&po);
  po.PrintUsage();
  po.Read(argc, argv);
  std::cout << config.ToString() << "\n";
}
