// Copyright (c)  2026  Xiaomi Corporation
// Web-specific TTS model loading.
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
JSObject configToJs(sherpa_onnx.OfflineTtsConfig cfg) {
  final jsConfig = JSObject();
  final model = JSObject();
  final m = cfg.model;

  // Only include model types that are actually configured.
  if (m.vits.model.isNotEmpty) {
    final vits = JSObject();
    vits['model'] = m.vits.model.toJS;
    vits['tokens'] = m.vits.tokens.toJS;
    if (m.vits.dataDir.isNotEmpty) vits['dataDir'] = m.vits.dataDir.toJS;
    if (m.vits.lexicon.isNotEmpty) vits['lexicon'] = m.vits.lexicon.toJS;
    vits['noiseScale'] = m.vits.noiseScale.toJS;
    vits['noiseScaleW'] = m.vits.noiseScaleW.toJS;
    vits['lengthScale'] = m.vits.lengthScale.toJS;
    model['vits'] = vits;
  }

  if (m.kokoro.model.isNotEmpty) {
    final kokoro = JSObject();
    kokoro['model'] = m.kokoro.model.toJS;
    kokoro['tokens'] = m.kokoro.tokens.toJS;
    if (m.kokoro.dataDir.isNotEmpty) kokoro['dataDir'] = m.kokoro.dataDir.toJS;
    if (m.kokoro.lexicon.isNotEmpty) kokoro['lexicon'] = m.kokoro.lexicon.toJS;
    if (m.kokoro.voices.isNotEmpty) kokoro['voices'] = m.kokoro.voices.toJS;
    model['kokoro'] = kokoro;
  }

  if (m.kitten.model.isNotEmpty) {
    final kitten = JSObject();
    kitten['model'] = m.kitten.model.toJS;
    kitten['tokens'] = m.kitten.tokens.toJS;
    if (m.kitten.dataDir.isNotEmpty) kitten['dataDir'] = m.kitten.dataDir.toJS;
    if (m.kitten.voices.isNotEmpty) kitten['voices'] = m.kitten.voices.toJS;
    model['kitten'] = kitten;
  }

  if (m.matcha.acousticModel.isNotEmpty) {
    final matcha = JSObject();
    matcha['acousticModel'] = m.matcha.acousticModel.toJS;
    matcha['vocoder'] = m.matcha.vocoder.toJS;
    matcha['tokens'] = m.matcha.tokens.toJS;
    if (m.matcha.dataDir.isNotEmpty) matcha['dataDir'] = m.matcha.dataDir.toJS;
    model['matcha'] = matcha;
  }

  if (m.pocket.lmFlow.isNotEmpty) {
    final pocket = JSObject();
    pocket['lmFlow'] = m.pocket.lmFlow.toJS;
    pocket['lmMain'] = m.pocket.lmMain.toJS;
    pocket['encoder'] = m.pocket.encoder.toJS;
    pocket['decoder'] = m.pocket.decoder.toJS;
    pocket['textConditioner'] = m.pocket.textConditioner.toJS;
    pocket['vocabJson'] = m.pocket.vocabJson.toJS;
    pocket['tokenScoresJson'] = m.pocket.tokenScoresJson.toJS;
    model['pocket'] = pocket;
  }

  if (m.supertonic.durationPredictor.isNotEmpty) {
    final supertonic = JSObject();
    supertonic['durationPredictor'] = m.supertonic.durationPredictor.toJS;
    supertonic['textEncoder'] = m.supertonic.textEncoder.toJS;
    supertonic['vectorEstimator'] = m.supertonic.vectorEstimator.toJS;
    supertonic['vocoder'] = m.supertonic.vocoder.toJS;
    supertonic['ttsJson'] = m.supertonic.ttsJson.toJS;
    supertonic['unicodeIndexer'] = m.supertonic.unicodeIndexer.toJS;
    supertonic['voiceStyle'] = m.supertonic.voiceStyle.toJS;
    model['supertonic'] = supertonic;
  }

  if (m.zipvoice.encoder.isNotEmpty) {
    final zipvoice = JSObject();
    zipvoice['tokens'] = m.zipvoice.tokens.toJS;
    zipvoice['encoder'] = m.zipvoice.encoder.toJS;
    zipvoice['decoder'] = m.zipvoice.decoder.toJS;
    zipvoice['vocoder'] = m.zipvoice.vocoder.toJS;
    if (m.zipvoice.dataDir.isNotEmpty) zipvoice['dataDir'] = m.zipvoice.dataDir.toJS;
    if (m.zipvoice.lexicon.isNotEmpty) zipvoice['lexicon'] = m.zipvoice.lexicon.toJS;
    model['zipvoice'] = zipvoice;
  }

  model['numThreads'] = m.numThreads.toJS;
  model['debug'] = m.debug.toJS;
  model['provider'] = m.provider.toJS;

  jsConfig['model'] = model;
  if (cfg.ruleFsts.isNotEmpty) jsConfig['ruleFsts'] = cfg.ruleFsts.toJS;
  if (cfg.ruleFars.isNotEmpty) jsConfig['ruleFars'] = cfg.ruleFars.toJS;
  jsConfig['maxNumSentences'] = cfg.maxNumSenetences.toJS;
  jsConfig['silenceScale'] = cfg.silenceScale.toJS;

  return jsConfig;
}

Future<Uint8List> _loadAsset(String assetPath) async {
  final data = await rootBundle.load(assetPath);
  return data.buffer.asUint8List();
}
