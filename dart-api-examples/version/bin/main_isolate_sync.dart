// Copyright (c)  2026  Xiaomi Corporation
//
// Example: initialization in isolates using initBindings() (sync).
//
// IMPORTANT: Each isolate has its own FFI binding state.
// You MUST call initBindings() or initBindingsAsync() in every isolate
// that uses sherpa-onnx APIs. Calling it in one isolate does NOT make
// sherpa-onnx available in other isolates.
import 'dart:isolate';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void worker(SendPort sendPort) {
  // Each isolate must initialize sherpa-onnx independently.
  sherpa_onnx.initBindings();

  final result = {
    'isolate': 'worker',
    'version': sherpa_onnx.getVersion(),
    'git_date': sherpa_onnx.getGitDate(),
    'git_sha1': sherpa_onnx.getGitSha1(),
    'onnxruntime': sherpa_onnx.getOnnxruntimeVersion(),
  };

  sendPort.send(result);
}

void main() async {
  // Initialize in the main isolate.
  sherpa_onnx.initBindings();

  print('=== Main isolate ===');
  print('version: ${sherpa_onnx.getVersion()}');
  print('git date: ${sherpa_onnx.getGitDate()}');
  print('git sha1: ${sherpa_onnx.getGitSha1()}');
  print('onnxruntime: ${sherpa_onnx.getOnnxruntimeVersion()}');

  // Spawn a worker isolate and receive its result.
  final receivePort = ReceivePort();
  await Isolate.spawn(worker, receivePort.sendPort);

  final result = await receivePort.first;
  print('\n=== Worker isolate ===');
  print('version: ${result['version']}');
  print('git date: ${result['git_date']}');
  print('git sha1: ${result['git_sha1']}');
  print('onnxruntime: ${result['onnxruntime']}');
}
