// Copyright (c)  2026  Xiaomi Corporation
//
// Example: async initialization using initBindingsAsync().
//
// initBindingsAsync() works on all platforms (Flutter, Dart CLI, Web).
// It auto-resolves the native library path on Dart CLI and loads WASM on web.
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main() async {
  await sherpa_onnx.initBindingsAsync();

  print('sherpa-onnx version: ${sherpa_onnx.getVersion()}');
  print('sherpa-onnx git sha1: ${sherpa_onnx.getGitSha1()}');
  print('sherpa-onnx git date: ${sherpa_onnx.getGitDate()}');
  print('onnxruntime version: ${sherpa_onnx.getOnnxruntimeVersion()}');
}
