// Copyright (c)  2026  Xiaomi Corporation
// Web implementation of OfflineTTS using dart:js_interop.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import '../tts_config.dart';
import 'init.dart';

export '../tts_config.dart';

/// Offline text-to-speech engine (web implementation).
class OfflineTts {
  OfflineTts.fromPtr({required this.ptr, required this.config});
  OfflineTts._({required this.ptr, required this.config});

  /// Create a TTS instance using the JS wrapper.
  factory OfflineTts(OfflineTtsConfig config) {
    final m = getModule();
    // createOfflineTts is a global function defined by sherpa-onnx-tts.js.
    final createFn =
        globalContext.getProperty('createOfflineTts'.toJS) as JSFunction?;
    if (createFn == null) {
      throw StateError('createOfflineTts not found. Is sherpa-onnx-tts.js loaded?');
    }

    // Convert Dart config to JS object.
    final jsConfig = _configToJs(config);
    final handle = createFn.callAsFunction(null, m, jsConfig);

    if (handle == null) {
      throw Exception('Failed to create OfflineTts on web');
    }

    return OfflineTts._(ptr: handle, config: config);
  }

  void free() {
    if (_freed) return;
    final m = getModule();
    final destroyFn = m.getProperty('_SherpaOnnxDestroyOfflineTts'.toJS) as JSFunction?;
    destroyFn?.callAsFunction(null, ptr);
    _freed = true;
  }

  GeneratedAudio generate({required String text, int sid = 0, double speed = 1.0}) {
    return generateWithConfig(
      text: text,
      config: OfflineTtsGenerationConfig(sid: sid, speed: speed),
    );
  }

  GeneratedAudio generateWithCallback({
    required String text,
    int sid = 0,
    double speed = 1.0,
    required int Function(Float32List samples) callback,
  }) {
    return generateWithConfig(
      text: text,
      config: OfflineTtsGenerationConfig(sid: sid, speed: speed),
      onProgress: (samples, progress) => callback(samples),
    );
  }

  GeneratedAudio generateWithConfig({
    required String text,
    required OfflineTtsGenerationConfig config,
    int Function(Float32List samples, double progress)? onProgress,
  }) {
    final m = getModule();
    final handle = ptr as JSObject;

    // Build the genConfig JS object.
    final genConfig = JSObject();
    genConfig['silenceScale'] = config.silenceScale.toJS;
    genConfig['speed'] = config.speed.toJS;
    genConfig['sid'] = config.sid.toJS;
    genConfig['numSteps'] = config.numSteps.toJS;

    // Reference audio for voice cloning (e.g. Pocket TTS).
    if (config.referenceAudio != null && config.referenceAudio!.isNotEmpty) {
      // Convert Float32List to a JS Float32Array via the ArrayBuffer constructor.
      final arrayBuffer = config.referenceAudio!.buffer.toJS;
      final float32ArrayCtor =
          globalContext.getProperty('Float32Array'.toJS) as JSFunction;
      genConfig['referenceAudio'] =
          float32ArrayCtor.callAsConstructor(arrayBuffer);
      genConfig['referenceSampleRate'] = config.referenceSampleRate.toJS;
      genConfig['referenceText'] = config.referenceText.toJS;
    }

    // Extra model-specific attributes.
    if (config.extra.isNotEmpty) {
      final extraObj = JSObject();
      for (final entry in config.extra.entries) {
        extraObj[entry.key] = (entry.value is String
            ? (entry.value as String).toJS
            : entry.value is int
                ? (entry.value as int).toJS
                : (entry.value as double).toJS) as JSAny;
      }
      genConfig['extra'] = extraObj;
    }

    if (onProgress != null) {
      // Create a JS callback that calls the Dart callback.
      genConfig['callback'] = (JSAny samplesPtr, JSAny n, JSAny progress, JSAny arg) {
        // The JS wrapper passes a Float32Array as samplesPtr.
        final samples = (samplesPtr as JSFloat32Array).toDart;
        return onProgress(samples, (progress as JSNumber).toDartDouble).toJS;
      }.toJS;
    }

    // Call the JS wrapper's generateWithConfig method.
    final generateFn = handle.getProperty('generateWithConfig'.toJS) as JSFunction?;
    if (generateFn == null) {
      throw StateError('generateWithConfig not found on OfflineTts instance');
    }

    final result = generateFn.callAsFunction(handle, text.toJS, genConfig) as JSObject;
    final samples = (result.getProperty('samples'.toJS) as JSFloat32Array).toDart;
    final sampleRate = (result.getProperty('sampleRate'.toJS) as JSNumber).toDartInt;

    return GeneratedAudio(samples: samples, sampleRate: sampleRate);
  }

  int get sampleRate {
    final handle = ptr as JSObject;
    final val = handle.getProperty('sampleRate'.toJS);
    return val is JSNumber ? val.toDartInt : 0;
  }

  int get numSpeakers {
    final handle = ptr as JSObject;
    final val = handle.getProperty('numSpeakers'.toJS);
    return val is JSNumber ? val.toDartInt : 0;
  }

  dynamic ptr;
  OfflineTtsConfig config;
  bool _freed = false;
}

/// Convert OfflineTtsConfig to a JS object for the JS wrapper.
JSObject _configToJs(OfflineTtsConfig config) {
  final jsConfig = JSObject();

  // model config
  final model = JSObject();
  final vits = JSObject();
  vits['model'] = config.model.vits.model.toJS;
  vits['lexicon'] = config.model.vits.lexicon.toJS;
  vits['tokens'] = config.model.vits.tokens.toJS;
  vits['dataDir'] = config.model.vits.dataDir.toJS;
  vits['noiseScale'] = config.model.vits.noiseScale.toJS;
  vits['noiseScaleW'] = config.model.vits.noiseScaleW.toJS;
  vits['lengthScale'] = config.model.vits.lengthScale.toJS;
  model['vits'] = vits;

  final matcha = JSObject();
  matcha['acousticModel'] = config.model.matcha.acousticModel.toJS;
  matcha['vocoder'] = config.model.matcha.vocoder.toJS;
  matcha['lexicon'] = config.model.matcha.lexicon.toJS;
  matcha['tokens'] = config.model.matcha.tokens.toJS;
  matcha['dataDir'] = config.model.matcha.dataDir.toJS;
  matcha['noiseScale'] = config.model.matcha.noiseScale.toJS;
  matcha['lengthScale'] = config.model.matcha.lengthScale.toJS;
  model['matcha'] = matcha;

  final kokoro = JSObject();
  kokoro['model'] = config.model.kokoro.model.toJS;
  kokoro['voices'] = config.model.kokoro.voices.toJS;
  kokoro['tokens'] = config.model.kokoro.tokens.toJS;
  kokoro['dataDir'] = config.model.kokoro.dataDir.toJS;
  kokoro['lengthScale'] = config.model.kokoro.lengthScale.toJS;
  kokoro['lexicon'] = config.model.kokoro.lexicon.toJS;
  kokoro['lang'] = config.model.kokoro.lang.toJS;
  model['kokoro'] = kokoro;

  final kitten = JSObject();
  kitten['model'] = config.model.kitten.model.toJS;
  kitten['voices'] = config.model.kitten.voices.toJS;
  kitten['tokens'] = config.model.kitten.tokens.toJS;
  kitten['dataDir'] = config.model.kitten.dataDir.toJS;
  kitten['lengthScale'] = config.model.kitten.lengthScale.toJS;
  model['kitten'] = kitten;

  final zipvoice = JSObject();
  zipvoice['tokens'] = config.model.zipvoice.tokens.toJS;
  zipvoice['encoder'] = config.model.zipvoice.encoder.toJS;
  zipvoice['decoder'] = config.model.zipvoice.decoder.toJS;
  zipvoice['vocoder'] = config.model.zipvoice.vocoder.toJS;
  zipvoice['dataDir'] = config.model.zipvoice.dataDir.toJS;
  zipvoice['lexicon'] = config.model.zipvoice.lexicon.toJS;
  zipvoice['featScale'] = config.model.zipvoice.featScale.toJS;
  zipvoice['tShift'] = config.model.zipvoice.tShift.toJS;
  zipvoice['targetRms'] = config.model.zipvoice.targetRms.toJS;
  zipvoice['guidanceScale'] = config.model.zipvoice.guidanceScale.toJS;
  model['zipvoice'] = zipvoice;

  final pocket = JSObject();
  pocket['lmFlow'] = config.model.pocket.lmFlow.toJS;
  pocket['lmMain'] = config.model.pocket.lmMain.toJS;
  pocket['encoder'] = config.model.pocket.encoder.toJS;
  pocket['decoder'] = config.model.pocket.decoder.toJS;
  pocket['textConditioner'] = config.model.pocket.textConditioner.toJS;
  pocket['vocabJson'] = config.model.pocket.vocabJson.toJS;
  pocket['tokenScoresJson'] = config.model.pocket.tokenScoresJson.toJS;
  pocket['voiceEmbeddingCacheCapacity'] =
      config.model.pocket.voiceEmbeddingCacheCapacity.toJS;
  model['pocket'] = pocket;

  final supertonic = JSObject();
  supertonic['durationPredictor'] = config.model.supertonic.durationPredictor.toJS;
  supertonic['textEncoder'] = config.model.supertonic.textEncoder.toJS;
  supertonic['vectorEstimator'] = config.model.supertonic.vectorEstimator.toJS;
  supertonic['vocoder'] = config.model.supertonic.vocoder.toJS;
  supertonic['ttsJson'] = config.model.supertonic.ttsJson.toJS;
  supertonic['unicodeIndexer'] = config.model.supertonic.unicodeIndexer.toJS;
  supertonic['voiceStyle'] = config.model.supertonic.voiceStyle.toJS;
  model['supertonic'] = supertonic;

  model['numThreads'] = config.model.numThreads.toJS;
  model['debug'] = config.model.debug.toJS;
  model['provider'] = config.model.provider.toJS;

  jsConfig['offlineTtsModelConfig'] = model;
  jsConfig['ruleFsts'] = config.ruleFsts.toJS;
  jsConfig['ruleFars'] = config.ruleFars.toJS;
  jsConfig['maxNumSentences'] = config.maxNumSenetences.toJS;
  jsConfig['silenceScale'] = config.silenceScale.toJS;

  return jsConfig;
}
