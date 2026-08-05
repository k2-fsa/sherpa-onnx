// Copyright (c)  2026  Xiaomi Corporation
// Web-specific TTS model loading.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './model_config.dart';

/// Flat config for the worker (matches the worker's expected format).
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
  final double noiseScale;
  final double noiseScaleW;
  final double lengthScale;

  const TtsModelConfig({
    required this.modelPath,
    required this.tokensPath,
    this.dataDir = '',
    this.lexicon = '',
    this.ruleFsts = '',
    this.ruleFars = '',
    this.voices = '',
    this.isKitten = false,
    this.numThreads = 2,
    this.debug = true,
    this.provider = 'cpu',
    this.noiseScale = 0.667,
    this.noiseScaleW = 0.8,
    this.lengthScale = 1.0,
  });
}

/// Prepare model config from shared model selection.
Future<TtsModelConfig> prepareModelConfig() async {
  final cfg = selectedTtsConfig;
  final m = cfg.model;

  // Debug logging.
  print('[model_web] prepareModelConfig()');
  print('[model_web]   selectedModelIndex: $selectedModelIndex');
  print('[model_web]   vits.model: ${m.vits.model}');
  print('[model_web]   vits.tokens: ${m.vits.tokens}');
  print('[model_web]   vits.dataDir: ${m.vits.dataDir}');
  print('[model_web]   vits.lexicon: ${m.vits.lexicon}');
  print('[model_web]   numThreads: ${m.numThreads}');
  print('[model_web]   debug: ${m.debug}');
  print('[model_web]   provider: ${m.provider}');
  print('[model_web]   ruleFsts: ${cfg.ruleFsts}');
  print('[model_web]   ruleFars: ${cfg.ruleFars}');

  // Determine which model type is active and extract paths.
  if (m.vits.model.isNotEmpty) {
    return TtsModelConfig(
      modelPath: m.vits.model,
      tokensPath: m.vits.tokens,
      dataDir: m.vits.dataDir,
      lexicon: m.vits.lexicon,
      numThreads: m.numThreads,
      debug: m.debug,
      provider: m.provider,
      noiseScale: m.vits.noiseScale,
      noiseScaleW: m.vits.noiseScaleW,
      lengthScale: m.vits.lengthScale,
    );
  }
  if (m.kokoro.model.isNotEmpty) {
    return TtsModelConfig(
      modelPath: m.kokoro.model,
      tokensPath: m.kokoro.tokens,
      dataDir: m.kokoro.dataDir,
      lexicon: m.kokoro.lexicon,
      voices: m.kokoro.voices,
      numThreads: m.numThreads,
      debug: m.debug,
      provider: m.provider,
    );
  }
  if (m.kitten.model.isNotEmpty) {
    return TtsModelConfig(
      modelPath: m.kitten.model,
      tokensPath: m.kitten.tokens,
      dataDir: m.kitten.dataDir,
      voices: m.kitten.voices,
      isKitten: true,
      numThreads: m.numThreads,
      debug: m.debug,
      provider: m.provider,
    );
  }

  // Default fallback.
  return TtsModelConfig(
    modelPath: m.vits.model,
    tokensPath: m.vits.tokens,
    numThreads: m.numThreads,
    debug: m.debug,
    provider: m.provider,
  );
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

/// Create a TTS instance from config (web — not used directly).
/// On web, the worker creates the TTS instance.
sherpa_onnx.OfflineTts createTtsFromConfig(TtsModelConfig cfg) {
  throw UnsupportedError('createTtsFromConfig is not available on web');
}

Future<Uint8List> _loadAsset(String assetPath) async {
  final data = await rootBundle.load(assetPath);
  return data.buffer.asUint8List();
}
