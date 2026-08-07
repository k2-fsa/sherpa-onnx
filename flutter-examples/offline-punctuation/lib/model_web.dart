// Copyright (c)  2026  Xiaomi Corporation
// Web-specific punctuation model loading.
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './model_config.dart';

/// Prepare model config for web (paths relative to WASM FS).
Future<sherpa_onnx.OfflinePunctuationConfig> prepareModelConfig() async {
  return punctConfig;
}

/// Load model file bytes from Flutter assets.
Future<Map<String, Uint8List>> loadModelFileBytes() async {
  final assetManifest = await AssetManifest.loadFromAssetBundle(rootBundle);
  final allAssets = assetManifest.listAssets();

  final modelAssets = allAssets.where((a) {
    return a.contains(modelDir);
  }).toList();

  final result = <String, Uint8List>{};
  for (final asset in modelAssets) {
    final bytes = await _loadAsset(asset);
    final relativePath = asset.replaceFirst('assets/', '');
    result[relativePath] = bytes;
  }
  return result;
}

/// Convert OfflinePunctuationConfig to JS config for the worker.
JSObject configToJs(sherpa_onnx.OfflinePunctuationConfig cfg) {
  final jsonStr = jsonEncode(cfg.toJson());
  final jsonObj = globalContext.getProperty('JSON'.toJS) as JSObject;
  final jsonParse = jsonObj.getProperty('parse'.toJS) as JSFunction;
  return jsonParse.callAsFunction(jsonObj, jsonStr.toJS) as JSObject;
}

Future<Uint8List> _loadAsset(String assetPath) async {
  final data = await rootBundle.load(assetPath);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}
