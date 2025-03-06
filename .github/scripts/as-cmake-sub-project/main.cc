#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/parse-options.h"

int main(int32_t argc, char *argv[]) {
  sherpa_onnx::ParseOptions po("help info");
  sherpa_onnx::OfflineRecognizerConfig config;
  config.Register(&po);
  po.PrintUsage();
  return 0;
}
