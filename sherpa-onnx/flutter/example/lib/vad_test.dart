// Copyright (c)  2024  Xiaomi Corporation
import 'dart:typed_data';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './utils.dart';

Future<void> testVad() async {
  final src = 'assets/silero_vad.onnx';
  final modelPath = await copyAssetFile(src: src, dst: 'silero_vad.onnx');

  final sileroVadConfig = sherpa_onnx.SileroVadModelConfig(model: modelPath);
  final config = sherpa_onnx.VadModelConfig(
    sileroVad: sileroVadConfig,
    numThreads: 1,
    debug: true,
  );

  final vad = sherpa_onnx.VoiceActivityDetector(
      config: config, bufferSizeInSeconds: 10);
  print('before vad.free(): ${vad.ptr}');
  vad.free();
  print('after vad.free(): ${vad.ptr}');

  final buffer = sherpa_onnx.CircularBuffer(capacity: 16000 * 2);

  final d = Float32List.fromList([0, 10, 20, 30]);
  buffer.push(d);
  assert(d.length == buffer.size, '${d.length} vs ${buffer.size}');

  final f = Float32List.fromList([-5, 100.25, 599]);
  buffer.push(f);

  assert(buffer.size == d.length + f.length);
  final g = buffer.get(startIndex: 0, n: 5);

  assert(g.length == 5);
  assert(g[0] == 0);
  assert(g[1] == 10);
  assert(g[2] == 20);
  assert(g[3] == 30);
  assert(g[4] == -5);

  assert(buffer.size == d.length + f.length);

  buffer.pop(3);
  assert(buffer.size == d.length + f.length - 3);

  final h = buffer.get(startIndex: buffer.head, n: 4);
  assert(h.length == 4);
  assert(h[0] == 30);
  assert(h[1] == -5);
  assert(h[2] == 100.25);
  assert(h[3] == 599);

  buffer.reset();

  assert(buffer.size == 0);
  assert(buffer.head == 0);

  print('before free: ${buffer.ptr}');
  buffer.free();
  print('after free: ${buffer.ptr}');
}
