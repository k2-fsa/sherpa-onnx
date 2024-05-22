import 'package:flutter/foundation.dart';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

Future<void> testVad() async {
  final buffer = sherpa_onnx.CircularBuffer(capacity: 16000 * 2);
  print('before free: ${buffer.ptr}');
  buffer.free();
  print('after free: ${buffer.ptr}');
}
