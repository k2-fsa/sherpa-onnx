// Copyright (c)  2026  Xiaomi Corporation
// Web-specific TTS model loading.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

/// Configuration for creating a TTS instance (web version).
/// On web, paths are relative to the WASM virtual filesystem.
class TtsModelConfig {
  final String modelPath;
  final String tokensPath;
  final String dataDir;
  final String lexicon;
  final String ruleFsts;
  final String ruleFars;
  final String voices;
  final bool isKitten;
  final int numThreads;
  final bool debug;
  final String provider;

  const TtsModelConfig({
    required this.modelPath,
    required this.tokensPath,
    this.dataDir = '',
    this.lexicon = '',
    this.ruleFsts = '',
    this.ruleFars = '',
    this.voices = '',
    this.isKitten = false,
    this.numThreads = 1,
    this.debug = true,
    this.provider = 'cpu',
  });
}

/// Prepare model config: load model assets from Flutter bundle.
Future<TtsModelConfig> prepareModelConfig() async {
  final modelDir = 'vits-piper-en_US-amy-low';
  return TtsModelConfig(
    modelPath: '$modelDir/en_US-amy-low.onnx',
    tokensPath: '$modelDir/tokens.txt',
    dataDir: '$modelDir/espeak-ng-data',
  );
}

/// Load model file bytes from Flutter assets.
/// Returns a map of {relativePath: bytes} for sending to a Web Worker.
Future<Map<String, Uint8List>> loadModelFileBytes() async {
  final assetManifest = await AssetManifest.loadFromAssetBundle(rootBundle);
  final allAssets = assetManifest.listAssets();

  final modelAssets =
      allAssets.where((a) => a.contains('vits-piper-en_US-amy-low')).toList();

  final result = <String, Uint8List>{};
  for (final asset in modelAssets) {
    final bytes = await _loadAsset(asset);
    final relativePath = asset.replaceFirst('assets/', '');
    result[relativePath] = bytes;
  }
  return result;
}

/// Create a TTS instance from config (web version).
sherpa_onnx.OfflineTts createTtsFromConfig(TtsModelConfig cfg) {
  final vits = sherpa_onnx.OfflineTtsVitsModelConfig(
    model: cfg.modelPath,
    tokens: cfg.tokensPath,
    dataDir: cfg.dataDir,
  );

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    vits: vits,
    numThreads: cfg.numThreads,
    debug: cfg.debug,
    provider: cfg.provider,
  );

  final config = sherpa_onnx.OfflineTtsConfig(
    model: modelConfig,
    maxNumSenetences: 1,
  );

  return sherpa_onnx.OfflineTts(config);
}

// ── WASM FS helpers ──────────────────────────────────────────────────────

JSObject _getFS() {
  final module = globalContext.getProperty('Module'.toJS) as JSObject?;
  if (module != null) {
    final fs = module.getProperty('FS'.toJS) as JSObject?;
    if (fs != null) return fs;
  }
  final globalFs = globalContext.getProperty('FS'.toJS) as JSObject?;
  if (globalFs != null) return globalFs;
  throw StateError('FS not found on Module.');
}

void _writeToWasmFS(String path, Uint8List data) {
  final fs = _getFS();
  final fn = fs.getProperty('writeFile'.toJS) as JSFunction;
  fn.callAsFunction(fs, path.toJS, data.toJS);
}

void _mkdirWasmFS(String path) {
  final fs = _getFS();
  final parts = path.split('/');
  String current = '';
  for (final part in parts) {
    if (part.isEmpty) continue;
    current = '$current/$part';
    try {
      final fn = fs.getProperty('mkdir'.toJS) as JSFunction;
      fn.callAsFunction(fs, current.toJS);
    } catch (_) {}
  }
}

Future<Uint8List> _loadAsset(String assetPath) async {
  final data = await rootBundle.load(assetPath);
  return data.buffer.asUint8List();
}
