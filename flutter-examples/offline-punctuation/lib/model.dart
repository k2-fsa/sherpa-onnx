// Copyright (c)  2026  Xiaomi Corporation
import "dart:io";

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './model_config.dart';

/// Resolve a relative path to absolute.
String _abs(String base, String relative) =>
    relative.isEmpty ? '' : p.join(base, relative);

/// Prepare model config: copy assets to disk and resolve all paths.
Future<sherpa_onnx.OfflinePunctuationConfig> prepareModelConfig() async {
  await _copyAllAssetFiles();

  final d = (await getApplicationSupportDirectory()).path;
  final cfg = punctConfig;

  return sherpa_onnx.OfflinePunctuationConfig(
    model: sherpa_onnx.OfflinePunctuationModelConfig(
      ctTransformer: _abs(d, cfg.model.ctTransformer),
      numThreads: cfg.model.numThreads,
      provider: cfg.model.provider,
      debug: cfg.model.debug,
    ),
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
