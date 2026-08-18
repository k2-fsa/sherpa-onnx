// Web Worker support for real-time VAD+ASR.
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;

import './model_web.dart' as m;
import './model_config.dart' as cfg;
import './vad_asr_manager_web.dart' show VadAsrSegment;

typedef OnReadyCallback = void Function();
typedef OnSpeechStateChangedCallback = void Function(bool isSpeaking);
typedef OnSegmentDetectedCallback = void Function(VadAsrSegment segment);
typedef OnErrorCallback = void Function(String message);

/// Manages a Web Worker for real-time VAD+ASR.
class VadAsrMicWorker {
  web.Worker? _worker;
  final OnReadyCallback onReady;
  final OnSpeechStateChangedCallback onSpeechStateChanged;
  final OnSegmentDetectedCallback onSegmentDetected;
  final OnErrorCallback onError;

  VadAsrMicWorker({
    required this.onReady,
    required this.onSpeechStateChanged,
    required this.onSegmentDetected,
    required this.onError,
  });

  /// Initialize the worker with VAD and ASR configs.
  Future<void> init({
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) async {
    final vadModelFiles = await m.loadModelFileBytes();
    final vadConfig = await m.prepareModelConfig();

    // Load ASR model files.
    final asrModelDir = cfg.selectedAsrModel.assetFiles[0]
        .substring(0, cfg.selectedAsrModel.assetFiles[0].lastIndexOf('/'));
    final asrModelFiles = <String, Uint8List>{};
    final assetManifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final allAssets = assetManifest.listAssets();
    for (final asset in allAssets) {
      if (asset.contains(asrModelDir)) {
        final bytes = await _loadAssetBytes(asset);
        final relativePath = asset.replaceFirst('assets/', '');
        asrModelFiles[relativePath] = bytes;
      }
    }

    // Build ASR config.
    final asrConfig = _buildAsrConfigForWeb();

    // Override VAD params.
    final jsVadConfig = m.configToJs(vadConfig);
    final sileroVad = jsVadConfig.getProperty('sileroVad'.toJS) as JSObject?;
    if (sileroVad != null) {
      sileroVad['threshold'] = threshold.toJS;
      sileroVad['minSilenceDuration'] = minSilenceDuration.toJS;
      sileroVad['minSpeechDuration'] = minSpeechDuration.toJS;
      sileroVad['maxSpeechDuration'] = maxSpeechDuration.toJS;
    }

    // Create Web Worker.
    _worker = web.Worker('vad-asr-worker.js'.toJS);

    _worker!.onmessage = (web.MessageEvent event) {
      _handleMessage(event);
    }.toJS;

    _worker!.onerror = (web.ErrorEvent event) {
      onError('Worker error: ${event.message}');
    }.toJS;

    // Load JS sources and WASM binary.
    final jsGlueSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.js');
    final vadJsSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-vad.js');
    final asrJsSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-asr.js');
    final wasmData = await _loadAssetBytes(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.wasm');

    // Build model file maps.
    final jsVadModelFiles = JSObject();
    for (final entry in vadModelFiles.entries) {
      jsVadModelFiles[entry.key] = Uint8List.fromList(entry.value).buffer.toJS;
    }
    final jsAsrModelFiles = JSObject();
    for (final entry in asrModelFiles.entries) {
      jsAsrModelFiles[entry.key] = entry.value.buffer.toJS;
    }

    // Send init message.
    final initMsg = JSObject();
    initMsg['type'] = 'init'.toJS;
    initMsg['jsGlueSource'] = jsGlueSource.toJS;
    initMsg['vadJsSource'] = vadJsSource.toJS;
    initMsg['asrJsSource'] = asrJsSource.toJS;
    initMsg['wasmBinary'] = wasmData.buffer.toJS;
    initMsg['vadModelFiles'] = jsVadModelFiles;
    initMsg['asrModelFiles'] = jsAsrModelFiles;
    initMsg['vadConfig'] = jsVadConfig;
    initMsg['asrConfig'] = asrConfig;
    _worker!.postMessage(initMsg);
  }

  /// Send audio samples to the worker for VAD+ASR processing.
  void acceptWaveform(Float32List samples) {
    final msg = JSObject();
    msg['type'] = 'acceptWaveform'.toJS;
    msg['samples'] = samples.buffer.toJS;
    _worker?.postMessage(msg);
  }

  /// Reset VAD state for a new recording session.
  void reset() {
    final msg = JSObject();
    msg['type'] = 'reset'.toJS;
    _worker?.postMessage(msg);
  }

  void dispose() {
    _worker?.terminate();
    _worker = null;
  }

  static Future<String> _loadAssetAsString(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    return utf8.decode(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
  }

  static Future<Uint8List> _loadAssetBytes(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    return data.buffer
        .asUint8List(data.offsetInBytes, data.lengthInBytes);
  }

  /// Build ASR config as a JS object for the worker.
  JSObject _buildAsrConfigForWeb() {
    final jsConfig = JSObject();

    // Feature config.
    final feat = JSObject();
    feat['sampleRate'] = 16000.toJS;
    feat['featureDim'] = 80.toJS;
    jsConfig['featConfig'] = feat;

    // Model config — only set the active model's fields.
    final modelConfig = JSObject();
    final modelDir = cfg.selectedAsrModel.assetFiles[0]
        .substring(0, cfg.selectedAsrModel.assetFiles[0].lastIndexOf('/'));

    switch (cfg.selectedModelIndex) {
      case 0: // Zipformer CTC
        final zipformerCtc = JSObject();
        zipformerCtc['model'] = '$modelDir/model.int8.onnx'.toJS;
        modelConfig['zipformerCtc'] = zipformerCtc;
        modelConfig['tokens'] = '$modelDir/tokens.txt'.toJS;
        modelConfig['modelingUnit'] = 'cjkchar'.toJS;
        break;
      case 1: // SenseVoice
        final senseVoice = JSObject();
        senseVoice['model'] = '$modelDir/model.int8.onnx'.toJS;
        senseVoice['language'] = 'auto'.toJS;
        modelConfig['senseVoice'] = senseVoice;
        modelConfig['tokens'] = '$modelDir/tokens.txt'.toJS;
        break;
      case 2: // Whisper
        final whisper = JSObject();
        whisper['encoder'] = '$modelDir/tiny.en-encoder.int8.onnx'.toJS;
        whisper['decoder'] = '$modelDir/tiny.en-decoder.int8.onnx'.toJS;
        modelConfig['whisper'] = whisper;
        modelConfig['tokens'] = '$modelDir/tiny.en-tokens.txt'.toJS;
        modelConfig['modelType'] = 'whisper'.toJS;
        break;
      case 3: // NeMo Parakeet TDT v2 (transducer)
      case 4: // NeMo Parakeet TDT v3 (transducer)
        final transducer = JSObject();
        transducer['encoder'] = '$modelDir/encoder.int8.onnx'.toJS;
        transducer['decoder'] = '$modelDir/decoder.int8.onnx'.toJS;
        transducer['joiner'] = '$modelDir/joiner.int8.onnx'.toJS;
        modelConfig['transducer'] = transducer;
        modelConfig['tokens'] = '$modelDir/tokens.txt'.toJS;
        modelConfig['modelType'] = 'nemo_transducer'.toJS;
        break;
      case 5: // Moonshine tiny en
        final moonshine = JSObject();
        moonshine['encoder'] = '$modelDir/encoder_model.ort'.toJS;
        moonshine['mergedDecoder'] = '$modelDir/decoder_model_merged.ort'.toJS;
        modelConfig['moonshine'] = moonshine;
        modelConfig['tokens'] = '$modelDir/tokens.txt'.toJS;
        break;
      case 6: // Qwen3 ASR
        final qwen3Asr = JSObject();
        qwen3Asr['convFrontend'] = '$modelDir/conv_frontend.onnx'.toJS;
        qwen3Asr['encoder'] = '$modelDir/encoder.int8.onnx'.toJS;
        qwen3Asr['decoder'] = '$modelDir/decoder.int8.onnx'.toJS;
        qwen3Asr['tokenizer'] = '$modelDir/tokenizer/tokenizer.json'.toJS;
        modelConfig['qwen3Asr'] = qwen3Asr;
        break;
      case 7: // FunASR Nano
        final funasrNano = JSObject();
        funasrNano['encoderAdaptor'] = '$modelDir/encoder_adaptor.int8.onnx'.toJS;
        funasrNano['llm'] = '$modelDir/llm.int8.onnx'.toJS;
        funasrNano['embedding'] = '$modelDir/embedding.int8.onnx'.toJS;
        funasrNano['tokenizer'] = '$modelDir/Qwen3-0.6B/tokenizer.json'.toJS;
        modelConfig['funasrNano'] = funasrNano;
        break;
      case 8: // FireRed ASR CTC
        final fireRedAsrCtc = JSObject();
        fireRedAsrCtc['model'] = '$modelDir/model.int8.onnx'.toJS;
        modelConfig['fireRedAsrCtc'] = fireRedAsrCtc;
        modelConfig['tokens'] = '$modelDir/tokens.txt'.toJS;
        break;
    }

    modelConfig['numThreads'] = 1.toJS;
    modelConfig['debug'] = 1.toJS;
    modelConfig['provider'] = 'cpu'.toJS;

    jsConfig['modelConfig'] = modelConfig;
    return jsConfig;
  }

  void _handleMessage(web.MessageEvent event) {
    final data = event.data! as JSObject;
    final type = (data.getProperty('type'.toJS)! as JSString).toDart;

    if (type == 'ready') {
      onReady();
    } else if (type == 'speechStateChanged') {
      final isSpeaking =
          (data.getProperty('isSpeaking'.toJS)! as JSBoolean).toDart;
      onSpeechStateChanged(isSpeaking);
    } else if (type == 'segmentDetected') {
      final index =
          (data.getProperty('index'.toJS)! as JSNumber).toDartInt;
      final start =
          (data.getProperty('start'.toJS)! as JSNumber).toDartDouble;
      final end =
          (data.getProperty('end'.toJS)! as JSNumber).toDartDouble;
      final text =
          (data.getProperty('text'.toJS) as JSString?)?.toDart ?? '';
      final elapsedSeconds =
          (data.getProperty('elapsedSeconds'.toJS) as JSNumber?)?.toDartDouble ?? 0;

      final samplesRaw = data.getProperty('samples'.toJS)!;
      Float32List samples;
      if (samplesRaw.isA<JSFloat32Array>()) {
        samples =
            Float32List.fromList((samplesRaw as JSFloat32Array).toDart);
      } else {
        final arr = samplesRaw as JSArray;
        final list = List<double>.filled(arr.length, 0);
        for (int j = 0; j < arr.length; j++) {
          list[j] = (arr[j] as JSNumber).toDartDouble;
        }
        samples = Float32List.fromList(list);
      }

      onSegmentDetected(VadAsrSegment(
        index: index,
        start: start,
        end: end,
        samples: samples,
        text: text,
        elapsedSeconds: elapsedSeconds,
      ));
    } else if (type == 'error') {
      final msg =
          (data.getProperty('message'.toJS)! as JSString).toDart;
      onError(msg);
    }
  }
}
