// Copyright (c)  2026  Xiaomi Corporation
import "dart:io";

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './model_config.dart';

String _abs(String base, String relative) =>
    relative.isEmpty ? '' : p.join(base, relative);

/// Prepare model config: copy assets to disk and resolve all paths.
Future<sherpa_onnx.VadModelConfig> prepareModelConfig() async {
  await _copyAllAssetFiles();

  final d = (await getApplicationSupportDirectory()).path;
  final cfg = defaultVadConfig;

  return sherpa_onnx.VadModelConfig(
    sileroVad: sherpa_onnx.SileroVadModelConfig(
      model: _abs(d, cfg.sileroVad.model),
      threshold: cfg.sileroVad.threshold,
      minSilenceDuration: cfg.sileroVad.minSilenceDuration,
      minSpeechDuration: cfg.sileroVad.minSpeechDuration,
      windowSize: cfg.sileroVad.windowSize,
      maxSpeechDuration: cfg.sileroVad.maxSpeechDuration,
    ),
    tenVad: sherpa_onnx.TenVadModelConfig(
      model: _abs(d, cfg.tenVad.model),
      threshold: cfg.tenVad.threshold,
      minSilenceDuration: cfg.tenVad.minSilenceDuration,
      minSpeechDuration: cfg.tenVad.minSpeechDuration,
      windowSize: cfg.tenVad.windowSize,
      maxSpeechDuration: cfg.tenVad.maxSpeechDuration,
    ),
    sampleRate: cfg.sampleRate,
    numThreads: cfg.numThreads,
    provider: cfg.provider,
    debug: cfg.debug,
  );
}

// ── Asset copy helpers ───────────────────────────────────────────────────

Future<void> _copyAllAssetFiles() async {
  final AssetManifest assetManifest =
      await AssetManifest.loadFromAssetBundle(rootBundle);
  final List<String> assets = assetManifest.listAssets();
  for (final src in assets) {
    final dst = _stripLeadingDirectory(src);
    await _copyAssetFile(src, dst);
  }
}

String _stripLeadingDirectory(String src, {int n = 1}) {
  return p.joinAll(p.split(src).sublist(n));
}

Future<String> _copyAssetFile(String src, [String? dst]) async {
  final Directory directory = await getApplicationSupportDirectory();
  if (dst == null) dst = p.basename(src);
  final target = p.join(directory.path, dst);
  bool exists = await File(target).exists();
  final data = await rootBundle.load(src);
  if (!exists || File(target).lengthSync() != data.lengthInBytes) {
    final List<int> bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await (await File(target).create(recursive: true)).writeAsBytes(bytes);
  }
  return target;
}
