// Copyright (c)  2024  Xiaomi Corporation
import "dart:io";

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './model_config.dart';

/// Configuration for creating a TTS instance.
/// All paths are resolved to absolute paths on disk.
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

/// Resolve a relative path to absolute.
String _abs(String base, String relative) =>
    relative.isEmpty ? '' : p.join(base, relative);

/// Prepare model config: copy assets to disk and resolve all paths.
Future<TtsModelConfig> prepareModelConfig() async {
  await _copyAllAssetFiles();

  final cfg = selectedTtsConfig;
  final m = cfg.model;
  final d = (await getApplicationSupportDirectory()).path;

  // Determine active model type and resolve paths.
  if (m.vits.model.isNotEmpty) {
    return TtsModelConfig(
      modelPath: _abs(d, m.vits.model),
      tokensPath: _abs(d, m.vits.tokens),
      dataDir: _abs(d, m.vits.dataDir),
      lexicon: m.vits.lexicon,
      ruleFsts: cfg.ruleFsts,
      ruleFars: cfg.ruleFars,
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
      modelPath: _abs(d, m.kokoro.model),
      tokensPath: _abs(d, m.kokoro.tokens),
      dataDir: _abs(d, m.kokoro.dataDir),
      lexicon: m.kokoro.lexicon,
      voices: m.kokoro.voices,
      ruleFsts: cfg.ruleFsts,
      ruleFars: cfg.ruleFars,
      numThreads: m.numThreads,
      debug: m.debug,
      provider: m.provider,
    );
  }
  if (m.kitten.model.isNotEmpty) {
    return TtsModelConfig(
      modelPath: _abs(d, m.kitten.model),
      tokensPath: _abs(d, m.kitten.tokens),
      dataDir: _abs(d, m.kitten.dataDir),
      voices: m.kitten.voices,
      isKitten: true,
      ruleFsts: cfg.ruleFsts,
      ruleFars: cfg.ruleFars,
      numThreads: m.numThreads,
      debug: m.debug,
      provider: m.provider,
    );
  }

  // Default.
  return TtsModelConfig(
    modelPath: _abs(d, m.vits.model),
    tokensPath: _abs(d, m.vits.tokens),
    numThreads: m.numThreads,
    debug: m.debug,
    provider: m.provider,
  );
}

/// Create an OfflineTts from a resolved config.
/// Can be called from any thread (no platform channels).
sherpa_onnx.OfflineTts createTtsFromConfig(TtsModelConfig cfg) {
  final vits = cfg.isKitten || cfg.voices.isNotEmpty
      ? sherpa_onnx.OfflineTtsVitsModelConfig()
      : sherpa_onnx.OfflineTtsVitsModelConfig(
          model: cfg.modelPath,
          lexicon: cfg.lexicon,
          tokens: cfg.tokensPath,
          dataDir: cfg.dataDir,
        );

  final kokoro = cfg.voices.isNotEmpty && !cfg.isKitten
      ? sherpa_onnx.OfflineTtsKokoroModelConfig(
          model: cfg.modelPath,
          voices: cfg.voices,
          tokens: cfg.tokensPath,
          dataDir: cfg.dataDir,
          lexicon: cfg.lexicon,
        )
      : sherpa_onnx.OfflineTtsKokoroModelConfig();

  final kitten = cfg.isKitten
      ? sherpa_onnx.OfflineTtsKittenModelConfig(
          model: cfg.modelPath,
          voices: cfg.voices,
          tokens: cfg.tokensPath,
          dataDir: cfg.dataDir,
        )
      : sherpa_onnx.OfflineTtsKittenModelConfig();

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    vits: vits,
    kokoro: kokoro,
    kitten: kitten,
    numThreads: cfg.numThreads,
    debug: cfg.debug,
    provider: cfg.provider,
  );

  final config = sherpa_onnx.OfflineTtsConfig(
    model: modelConfig,
    ruleFsts: cfg.ruleFsts,
    ruleFars: cfg.ruleFars,
    maxNumSenetences: 1,
  );

  return sherpa_onnx.OfflineTts(config);
}

/// Copy all Flutter assets to the app support directory.
/// Skips files that already exist with the correct size.
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
  if (dst == null) {
    dst = p.basename(src);
  }
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
