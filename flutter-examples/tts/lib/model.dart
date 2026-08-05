// Copyright (c)  2024  Xiaomi Corporation
import "dart:io";

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './model_config.dart';

/// Resolve a relative path to absolute.
String _abs(String base, String relative) =>
    relative.isEmpty ? '' : p.join(base, relative);

/// Resolve comma-separated relative paths to absolute.
String _absMulti(String base, String csv) =>
    csv.isEmpty ? '' : csv.split(',').map((f) => _abs(base, f.trim())).join(',');

/// Prepare model config: copy assets to disk and resolve all paths.
Future<sherpa_onnx.OfflineTtsConfig> prepareModelConfig() async {
  await _copyAllAssetFiles();

  final cfg = selectedTtsConfig;
  final d = (await getApplicationSupportDirectory()).path;
  final m = cfg.model;

  return sherpa_onnx.OfflineTtsConfig(
    model: sherpa_onnx.OfflineTtsModelConfig(
      vits: sherpa_onnx.OfflineTtsVitsModelConfig(
        model: _abs(d, m.vits.model),
        lexicon: _abs(d, m.vits.lexicon),
        tokens: _abs(d, m.vits.tokens),
        dataDir: _abs(d, m.vits.dataDir),
        noiseScale: m.vits.noiseScale,
        noiseScaleW: m.vits.noiseScaleW,
        lengthScale: m.vits.lengthScale,
      ),
      kokoro: sherpa_onnx.OfflineTtsKokoroModelConfig(
        model: _abs(d, m.kokoro.model),
        voices: _abs(d, m.kokoro.voices),
        tokens: _abs(d, m.kokoro.tokens),
        dataDir: _abs(d, m.kokoro.dataDir),
        lexicon: _absMulti(d, m.kokoro.lexicon),
        lang: m.kokoro.lang,
        lengthScale: m.kokoro.lengthScale,
      ),
      kitten: sherpa_onnx.OfflineTtsKittenModelConfig(
        model: _abs(d, m.kitten.model),
        voices: _abs(d, m.kitten.voices),
        tokens: _abs(d, m.kitten.tokens),
        dataDir: _abs(d, m.kitten.dataDir),
        lengthScale: m.kitten.lengthScale,
      ),
      matcha: sherpa_onnx.OfflineTtsMatchaModelConfig(
        acousticModel: _abs(d, m.matcha.acousticModel),
        vocoder: _abs(d, m.matcha.vocoder),
        tokens: _abs(d, m.matcha.tokens),
        dataDir: _abs(d, m.matcha.dataDir),
        lexicon: _abs(d, m.matcha.lexicon),
        noiseScale: m.matcha.noiseScale,
        lengthScale: m.matcha.lengthScale,
      ),
      pocket: sherpa_onnx.OfflineTtsPocketModelConfig(
        lmFlow: _abs(d, m.pocket.lmFlow),
        lmMain: _abs(d, m.pocket.lmMain),
        encoder: _abs(d, m.pocket.encoder),
        decoder: _abs(d, m.pocket.decoder),
        textConditioner: _abs(d, m.pocket.textConditioner),
        vocabJson: _abs(d, m.pocket.vocabJson),
        tokenScoresJson: _abs(d, m.pocket.tokenScoresJson),
        voiceEmbeddingCacheCapacity: m.pocket.voiceEmbeddingCacheCapacity,
      ),
      supertonic: sherpa_onnx.OfflineTtsSupertonicModelConfig(
        durationPredictor: _abs(d, m.supertonic.durationPredictor),
        textEncoder: _abs(d, m.supertonic.textEncoder),
        vectorEstimator: _abs(d, m.supertonic.vectorEstimator),
        vocoder: _abs(d, m.supertonic.vocoder),
        ttsJson: _abs(d, m.supertonic.ttsJson),
        unicodeIndexer: _abs(d, m.supertonic.unicodeIndexer),
        voiceStyle: _abs(d, m.supertonic.voiceStyle),
      ),
      zipvoice: sherpa_onnx.OfflineTtsZipVoiceModelConfig(
        tokens: _abs(d, m.zipvoice.tokens),
        encoder: _abs(d, m.zipvoice.encoder),
        decoder: _abs(d, m.zipvoice.decoder),
        vocoder: _abs(d, m.zipvoice.vocoder),
        dataDir: _abs(d, m.zipvoice.dataDir),
        lexicon: _abs(d, m.zipvoice.lexicon),
        featScale: m.zipvoice.featScale,
        tShift: m.zipvoice.tShift,
        targetRms: m.zipvoice.targetRms,
        guidanceScale: m.zipvoice.guidanceScale,
      ),
      numThreads: m.numThreads,
      debug: m.debug,
      provider: m.provider,
    ),
    ruleFsts: _absMulti(d, cfg.ruleFsts),
    ruleFars: _absMulti(d, cfg.ruleFars),
    maxNumSenetences: cfg.maxNumSenetences,
  );
}

/// Create an OfflineTts from a resolved config.
sherpa_onnx.OfflineTts createTtsFromConfig(sherpa_onnx.OfflineTtsConfig cfg) {
  return sherpa_onnx.OfflineTts(cfg);
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
