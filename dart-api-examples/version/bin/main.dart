// Copyright (c)  2026  Xiaomi Corporation

import 'package:sherpa_onnx/src/version.dart' as sherpa_onnx;

void main() {
  print('sherpa-onnx version: ${sherpa_onnx.getVersion()}');
  print('sherpa-onnx git sha1: ${sherpa_onnx.getGitSha1()}');
  print('sherpa-onnx git date: ${sherpa_onnx.getGitDate()}');
  print('onnxruntime version: ${sherpa_onnx.getOnnxruntimeVersion()}');
}
