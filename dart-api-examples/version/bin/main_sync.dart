// Copyright (c)  2026  Xiaomi Corporation
//
// Example: synchronous initialization using initBindings().
//
// initBindings() is simpler but does NOT support web.
// On Flutter and Dart CLI, it auto-resolves the native library path.
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main() {
  sherpa_onnx.initBindings();

  print('sherpa-onnx version: ${sherpa_onnx.getVersion()}');
  print('sherpa-onnx git sha1: ${sherpa_onnx.getGitSha1()}');
  print('sherpa-onnx git date: ${sherpa_onnx.getGitDate()}');
  print('onnxruntime version: ${sherpa_onnx.getOnnxruntimeVersion()}');
}
