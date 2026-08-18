// Copyright (c)  2026  Xiaomi Corporation
// Web-specific VAD model loading.
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './model_config.dart';

class ModelDirs {
  final String baseDir;
  final String asrModelDir;
  const ModelDirs({required this.baseDir, required this.asrModelDir});
}

/// On web, model paths are relative to the WASM FS.
Future<ModelDirs> prepareModelDirs() async {
  return const ModelDirs(baseDir: '', asrModelDir: '');
}

/// Prepare model config for web (paths relative to WASM FS).
Future<sherpa_onnx.VadModelConfig> prepareModelConfig() async {
  return defaultVadConfig;
}

/// Load model file bytes from Flutter assets.
Future<Map<String, Uint8List>> loadModelFileBytes() async {
  final assetManifest = await AssetManifest.loadFromAssetBundle(rootBundle);
  final allAssets = assetManifest.listAssets();

  // Filter assets by the selected model file name.
  final modelAssets = allAssets.where((a) {
    return a.contains(vadModelFile);
  }).toList();

  final result = <String, Uint8List>{};
  for (final asset in modelAssets) {
    final bytes = await _loadAsset(asset);
    final relativePath = asset.replaceFirst('assets/', '');
    result[relativePath] = bytes;
  }
  return result;
}

/// Convert VadModelConfig to JS config for the worker.
JSObject configToJs(sherpa_onnx.VadModelConfig cfg) {
  final jsonStr = jsonEncode(cfg.toJson());
  final jsonObj = globalContext.getProperty('JSON'.toJS) as JSObject;
  final jsonParse = jsonObj.getProperty('parse'.toJS) as JSFunction;
  return jsonParse.callAsFunction(jsonObj, jsonStr.toJS) as JSObject;
}

Future<Uint8List> _loadAsset(String assetPath) async {
  final data = await rootBundle.load(assetPath);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}
