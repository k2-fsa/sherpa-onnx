// Copyright (c)  2026  Xiaomi Corporation
// Web-specific TTS model loading.
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './model_config.dart';

/// Prepare model config for web (paths relative to WASM FS).
Future<sherpa_onnx.OfflineTtsConfig> prepareModelConfig() async {
  return selectedTtsConfig;
}

/// Load model file bytes from Flutter assets.
Future<Map<String, Uint8List>> loadModelFileBytes() async {
  final assetManifest = await AssetManifest.loadFromAssetBundle(rootBundle);
  final allAssets = assetManifest.listAssets();
  final cfg = selectedTtsConfig;

  // Collect all model directory names from the config.
  final modelDirs = <String>{};
  for (final path in [
    cfg.model.vits.model, cfg.model.vits.tokens, cfg.model.vits.dataDir,
    cfg.model.kokoro.model, cfg.model.kokoro.voices, cfg.model.kokoro.tokens,
    cfg.model.kitten.model, cfg.model.kitten.voices, cfg.model.kitten.tokens,
    cfg.model.matcha.acousticModel, cfg.model.matcha.vocoder,
    cfg.model.pocket.lmFlow, cfg.model.pocket.lmMain,
    cfg.model.supertonic.durationPredictor,
    cfg.model.zipvoice.encoder, cfg.model.zipvoice.tokens,
  ]) {
    if (path.isNotEmpty) {
      modelDirs.add(path.split('/').first);
    }
  }

  final modelAssets = allAssets.where((a) {
    for (final dir in modelDirs) {
      if (a.contains(dir)) return true;
    }
    return false;
  }).toList();

  final result = <String, Uint8List>{};
  for (final asset in modelAssets) {
    final bytes = await _loadAsset(asset);
    final relativePath = asset.replaceFirst('assets/', '');
    result[relativePath] = bytes;
  }
  return result;
}

/// Convert OfflineTtsConfig to JS config for the worker.
/// Uses toJson() from the config classes and converts to JSObject via JSON.
JSObject configToJs(sherpa_onnx.OfflineTtsConfig cfg) {
  final jsonStr = jsonEncode(cfg.toJson());
  final jsonObj = globalContext.getProperty('JSON'.toJS) as JSObject;
  final jsonParse = jsonObj.getProperty('parse'.toJS) as JSFunction;
  return jsonParse.callAsFunction(jsonObj, jsonStr.toJS) as JSObject;
}

Future<Uint8List> _loadAsset(String assetPath) async {
  final data = await rootBundle.load(assetPath);
  return data.buffer.asUint8List();
}
