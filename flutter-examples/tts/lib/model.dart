// Copyright (c)  2024  Xiaomi Corporation
import "dart:io";

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

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
  });
}

/// Prepare model config: copy assets to disk and resolve all paths.
/// This must be called on the main thread (uses platform channels).
Future<TtsModelConfig> prepareModelConfig() async {
  // Copy all asset files to the app support directory.
  await _copyAllAssetFiles();

  // Select your model here:
  final modelDir = 'vits-piper-en_US-amy-low';
  final modelName = 'en_US-amy-low.onnx';
  final dataDir = 'vits-piper-en_US-amy-low/espeak-ng-data';

  final Directory directory = await getApplicationSupportDirectory();

  final absModel = p.join(directory.path, modelDir, modelName);
  final absTokens = p.join(directory.path, modelDir, 'tokens.txt');
  final absDataDir = dataDir.isNotEmpty
      ? p.join(directory.path, dataDir)
      : '';

  return TtsModelConfig(
    modelPath: absModel,
    tokensPath: absTokens,
    dataDir: absDataDir,
    numThreads: 2,
    debug: true,
    provider: 'cpu',
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
